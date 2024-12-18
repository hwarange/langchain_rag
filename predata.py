import pandas as pd
import os
from pathlib import Path
import asyncio
import aiohttp
from dotenv import load_dotenv
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import warnings

# openpyxl 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

load_dotenv()

async def get_location_async(session, address, address_cache, pbar):
    """
    비동기로 주소의 위도/경도 좌표를 반환하는 함수
    """
    # 캐시된 결과가 있으면 반환
    if address in address_cache:
        pbar.update(1)
        return address_cache[address]
    
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {
        'Authorization': f'KakaoAK {os.environ.get("KAKAO_API_KEY")}'
    }
    params = {'query': address}
    
    try:
        async with session.get(url, headers=headers, params=params) as response:
            result = await response.json()
            
            if result.get('documents'):
                address_info = result['documents'][0]
                coords = {
                    'lat': float(address_info['y']),
                    'lng': float(address_info['x'])
                }
                address_cache[address] = coords
                pbar.update(1)
                return coords
    except Exception as e:
        print(f"Error getting coordinates for {address}: {str(e)}")
    
    pbar.update(1)
    return None

async def process_addresses(addresses):
    """
    여러 주소를 비동기로 처리하는 함수
    """
    address_cache = {}
    async with aiohttp.ClientSession() as session:
        tasks = []
        with tqdm(total=len(addresses), desc="Geocoding addresses") as pbar:
            for address in addresses:
                task = asyncio.create_task(
                    get_location_async(session, address, address_cache, pbar)
                )
                tasks.append(task)
                if len(tasks) >= 10:
                    await asyncio.sleep(0.5)
            
            results = await asyncio.gather(*tasks)
            return results

def process_address_data(file_path, process_id=None):
    """
    주소 데이터를 전처리하는 함수
    """
    try:
        df = pd.read_excel(file_path, header=12, skiprows=[13])
        
        if process_id is not None:
            print(f"Process {process_id} processing: {file_path}")
        
        print("Original columns:", df.columns.tolist())
        
        # 고유 ID 생성
        file_prefix = file_path.stem.split('_')[0]
        df['id'] = [f"{file_prefix}_{i:06d}" for i in range(len(df))]
        
        # 컬럼명 표준화
        column_mapping = {}
        for col in df.columns:
            col_str = str(col)
            if '시군구' in col_str:
                column_mapping[col] = 'district'
            elif '번지' in col_str or '지번' in col_str:
                column_mapping[col] = 'address_number'
            elif '단지명' in col_str:
                column_mapping[col] = 'complex_name'
            elif any(area in col_str for area in ['전용면적', '계약면적', '전용/연면적', '연면적']):
                column_mapping[col] = 'area_size'
            elif '거래금액' in col_str:
                column_mapping[col] = '거래금액'
            elif '계약년월' in col_str:
                column_mapping[col] = 'transaction_date'
            elif '층' in col_str:
                column_mapping[col] = 'floor'
            elif '보증금' in col_str and '종전' not in col_str:
                column_mapping[col] = '보증금'
            elif '월세' in col_str and '종전' not in col_str and '구분' not in col_str:
                column_mapping[col] = '월세'
        
        print("Column mapping:", column_mapping)
        df = df.rename(columns=column_mapping)
        
        # 불필요한 컬럼 제거
        drop_keywords = ['NO', '매수자', '매도자', '거래유형', 'main_number', 
                        'sub_number', 'address_number', '본번', '부번', 
                        '종전', '갱신요구권', '계약구분', '계약기간', '전월세구분']
        
        columns_to_drop = [col for col in df.columns 
                          if any(keyword in str(col) for keyword in drop_keywords)]
        
        df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # area_size 컬럼이 없는 경우 다른 면적 컬럼 찾기
        if 'area_size' not in df.columns:
            area_columns = [col for col in df.columns if '면적' in str(col)]
            if area_columns:
                df['area_size'] = df[area_columns[0]]
                df = df.drop(columns=area_columns[0])
        
        # 평수 계산 (1평 = 3.3058㎡)
        if 'area_size' in df.columns:
            df['평수'] = (df['area_size'] / 3.3058)
        
        # 거래 날짜 형식 변환
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'].astype(str), 
                                                  format='%Y%m').dt.strftime('%Y-%m')
        
        # 가격 데이터 처리
        def clean_price(value):
            if pd.isna(value):
                return value
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                return float(value.replace(',', ''))
            return value

        # 각 가격 컬럼 처리
        price_columns = ['거래금액', '보증금', '월세']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_price)
        
        # 면적 소수점 둘째자리까지 반올림
        if 'area_size' in df.columns:
            df['area_size'] = df['area_size'].round(2)
        
        # id 컬럼을 맨 앞으로 이동
        cols = df.columns.tolist()
        cols = ['id'] + [col for col in cols if col != 'id']
        df = df[cols]
        
        print("Processed columns:", df.columns.tolist())
        
        # 주소 결합 및 좌표 추가
        if 'district' in df.columns:
            if 'address_number' in df.columns:
                df['full_address'] = df['district'].astype(str).str.strip() + df['address_number'].astype(str).str.strip()
            else:
                df['full_address'] = df['district'].astype(str).str.strip()
            unique_addresses = df['full_address'].unique()
            coordinates = asyncio.run(process_addresses(unique_addresses))
            
            coordinates_dict = dict(zip(unique_addresses, coordinates))
            df['latitude'] = df['full_address'].map(lambda x: coordinates_dict.get(x, {}).get('lat'))
            df['longitude'] = df['full_address'].map(lambda x: coordinates_dict.get(x, {}).get('lng'))
        
        return df
    except Exception as e:
        print(f"Error in process {process_id} processing {file_path}: {str(e)}")
        return None

def process_all_files():
    """
    ./data 폴더의 모든 XLSX 파일을 병렬 처리하는 함수
    """
    data_dir = Path('./data')
    processed_file = data_dir / 'processed_data.csv'
    
    # 기존 처리된 데이터 로드
    existing_data = None
    existing_ids = set()
    if processed_file.exists():
        print("Loading existing processed data...")
        existing_data = pd.read_csv(processed_file)
        existing_ids = set(existing_data['id'])
        print(f"Found {len(existing_ids)} existing records")
    
    # 새로운 파일 처리
    file_paths = list(data_dir.glob('*.xlsx'))
    new_files = []
    
    for file_path in file_paths:
        file_prefix = file_path.stem.split('_')[0]
        # 파일의 첫 번째 ID를 확인 (예: "아파트_000000")
        test_id = f"{file_prefix}_000000"
        if test_id not in existing_ids:
            new_files.append(file_path)
    
    if not new_files:
        print("No new files to process")
        return existing_data
    
    print(f"Found {len(new_files)} new files to process")
    
    # CPU 코어 수를 기반으로 프로세스 풀 생성
    num_processes = min(mp.cpu_count(), len(new_files))
    print(f"Starting processing with {num_processes} processes...")
    
    with mp.Pool(num_processes) as pool:
        # 진행 상황을 표시하며 병렬 처리
        processed_data = list(tqdm(
            pool.imap(process_address_data, new_files),
            total=len(new_files),
            desc="Processing new files"
        ))
    
    # 성공적으로 처리된 데이터프레임만 선택
    processed_data = [df for df in processed_data if df is not None]
    
    if processed_data:
        print("\nCombining processed data...")
        if existing_data is not None:
            # 기존 데이터와 새로운 데이터 결합
            final_df = pd.concat([existing_data] + processed_data, ignore_index=True)
        else:
            final_df = pd.concat(processed_data, ignore_index=True)
        
        print("Removing duplicates...")
        final_df = final_df.drop_duplicates()
        
        print(f"Saving processed data to: {processed_file}")
        final_df.to_csv(processed_file, index=False, encoding='utf-8')
        
        # 새로 추가된 레코드 수 출력
        new_records = len(final_df) - (len(existing_data) if existing_data is not None else 0)
        print(f"Added {new_records} new records")
        
        return final_df
    
    return existing_data

if __name__ == "__main__":
    processed_df = process_all_files()
    if processed_df is not None:
        print("\nProcessing Summary:")
        print(f"Total records: {len(processed_df)}")
        print("\nColumn names:")
        for col in processed_df.columns:
            print(f"- {col}")
