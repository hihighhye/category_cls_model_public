import copy
import os
import re
import time
import openpyxl

import requests, json
from ast import literal_eval
import configparser
import json
import datetime as dt
from dateutil.parser import parse
# 아래 2개의 패키지 설치!!!!필수 !!!! (pymysql, sshtunnel)
import pymysql
import numpy as np
import pandas as pd
from tqdm import tqdm
# from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine, inspect
from sshtunnel import SSHTunnelForwarder
import traceback



SPLIT_CNT = 10000 # 한번에 2000건씩 제한(Elastic 서버에 영향때문)
CHUNKSIZE = 50000 # MART 데이터 iterator 건수

# MYSQL 접속 정보
BON_SERVER = {
    'ssh_host': 'server4.marketingtool.co.kr',
    'ssh_port': 22054,
    'ssh_username': 'dataqi', 
    'ssh_password': 'epdlxj24$',
    # 'db_name': 'lg_cns_22cat',
    'db_username': 'dataqi',
    'db_password': 'epdlxj24$',
}

# BON_SERVER = {
#     'ssh_host': 'server5.marketingtool.co.kr',
#     'ssh_port': 22102,
#     'ssh_username': 'dataqi', 
#     'ssh_password': 'epdlxj123',
#     # 'db_name': 'lg_cns_22cat',
#     'db_username': 'root',
#     'db_password': 'dataQI234!',
# }

AUTOMATCHING_SERVER = {
    'ssh_host': 'server5.marketingtool.co.kr',
    'ssh_port': 22102,
    'ssh_username': 'dataqi', 
    'ssh_password': 'epdlxj123',
    # 'db_name': 'lg_cns_22cat',
    'db_username': 'root',
    'db_password': 'dataQI234!',
}

SELF_SERVER = {
    'db_username': 'root',
    'db_password': 'dataQI234!',
}

ES_SERVER = {
    'url': 'http://dataqi:dataqi369@server5.marketingtool.co.kr:31400',
    'po_index': 'mass_po_hash_data_lgcns_mi_ecsm',
    'ro_index': 'mass_ro_hash_data_lgcns_mi_ecsm',
    'so_index': 'mass_so_hash_data_lgcns_mi',
    'split_size': 2000,
    'header': {'Content-Type': 'application/json'}
}



# Class 기준 사이트(프로젝트)별
class MysqlDb():
    def __init__(self, server_config):
        '''
        ○ 서버 및 DB 접근 관련 세팅
        '''
        # 서버 접속 정보
        self.server_config = server_config

        # 서버 및 데이터베이스에 접근
        if 'ssh_host' in server_config.keys():
            ssh_auto = SSHTunnelForwarder((server_config['ssh_host'], int(server_config['ssh_port'])),
                                        # 'ssh터널 host 주소'
                                        ssh_username=server_config['ssh_username'],
                                        ssh_password=server_config['ssh_password'],
                                        # ssh_pkey=ssh_auto_config['SSH_PKEY'],
                                        remote_bind_address=('127.0.0.1', 3306))
            ssh_auto.start()  # 원격 서버 접속 실행

            db_engine = create_engine(
                "mysql+pymysql://" + server_config['db_username'] + ":" + server_config['db_password'] + "@" \
                + '127.0.0.1' + ":" + str(ssh_auto.local_bind_port))  # 운영 스키마 ENGINE
        else:
            db_engine = create_engine(
                "mysql+pymysql://" + server_config['db_username'] + ":" + server_config['db_password'] + "@" \
                + '127.0.0.1' + ":" + '3306')  # 운영 스키마 ENGINE

        self.db_engine = db_engine

        # 아이템 마스터 불러오기
    def import_item_master(self):
        print('아이템 마스터 불러오기 함수 실행!!')

        item_master_path = r'Z:/home/dataqi/automatching/lg_cns_22cat/lg_cns_22cat_item_master.xlsx'
        im_file = openpyxl.load_workbook(item_master_path)

        item_master = pd.DataFrame()
        for sheet in im_file.get_sheet_names():
            group_category = sheet

            item_master_ps = pd.read_excel(item_master_path, dtype='str', sheet_name=sheet, \
                                        usecols=['group_category', 'top_node_title', 'middle_node_title',
                                                'bottom_node_title', 'below_node_title', 'replace_model', 'set_yn',
                                                'fixed_ctime'])
            item_master = pd.concat([item_master, item_master_ps], axis=0).reset_index(drop=True)
            
        # 공백처리
        item_master = item_master.fillna('')
        item_master = item_master.reset_index(drop=True)

        # 확정일자 필드 전처리('fixed_ctime')
        item_master.loc[:, 'fixed_ctime'] = item_master.loc[:, 'fixed_ctime'].map(lambda x: parse(x).strftime('%Y-%m-%d') if x != '' else '')

        print('아이템 마스터 불러오기 완료!!', item_master.head())

        return item_master

    ####################################################################################################################
    ### ○ 재정제 관련 함수
    ####################################################################################################################
    '''
    ○ 재정제 관련 함수 모음
    '''
    # IMPORT
    def import_data(self, db_name, tb_name, target_hashcodes=[], conditions=''):
        db_engine = self.db_engine

        sql = f"select * from {db_name}.{tb_name};"
        if target_hashcodes and conditions:
            target_format = '(' + str(target_hashcodes)[1:-1] + ')'
            sql = f"select * from {db_name}.{tb_name} where hashcode in {target_format} and {conditions};"
        elif target_hashcodes:
            target_format = '(' + str(target_hashcodes)[1:-1] + ')'
            sql = f"select * from {db_name}.{tb_name} where hashcode in {target_format};"
        elif conditions:
            sql = f"select * from {db_name}.{tb_name} where {conditions};"
        # print('실행 명령문 :', sql)
        data_iter = pd.read_sql(sql=sql, con=db_engine, index_col=None, chunksize=100000)
        print(f'데이터 iterator 생성완료 : {data_iter}')
        data = pd.DataFrame()
        for i, iter in enumerate(data_iter): # 10000 건씩 iterator
            print(i, iter)
            # if i > 10:
            #     break
            data = pd.concat([data, iter])

        for col in data.columns:
            data = data.astype({col : 'str'})

        print('데이터 가져오기 완료 :', data.head())

        return data

    # SAVE
    def save_data(self, data, db_name, tb_name):
        db_engine = self.db_engine
        print(f'스키마 : {db_name} || 데이터 : {tb_name} 적재 시작!!!')

        # 적재
        data.to_sql(name=tb_name, schema=db_name, con=db_engine, if_exists='append', index=False, chunksize=10000)
        print(f'스키마 : {db_name} || 데이터 : {tb_name} 적재 완료!!!')

        # print('실행명령문: ', f"select count(*) from {db_name}.{tb_name};")
        result = db_engine.execute(f"select count(*) from {db_name}.{tb_name};")
        # result_cnt = result.fetchall()
        result_cnt = result.first()[0]
        print(f'#'*30, f'적재 완료 후 데이터: {result_cnt}', '#'*30)

        return result_cnt
    
    def get_total_count(self, db_name, tb_name):
        db_engine = self.db_engine

        result = db_engine.execute(f"select count(*) from {db_name}.{tb_name};").scalar()
        result_cnt = result
        print(f'#'*30, f'현재 {tb_name} 총 데이터: {result_cnt}', '#'*30)

        return result_cnt

    # 해시코드 제거
    def delete_with_hashcodes(self, db_name, tb_name, target_hashcodes):
        db_engine = self.db_engine
        # TARGET_FLAG = '1' 삭제
        # delete_hashcodes =list(data[data['target_flag']=='1']['hashcode'])
        print(f'제거 대상 해시코드(재정제 대상) 건수 : {len(target_hashcodes)}')
        target_format = '(' + str(target_hashcodes)[1:-1] + ')'
        # 재정제 대상 해시코드 기존에 정제 완료된 테이블에서 제거
        db_engine.execute(f"delete from {db_name}.{tb_name} where hashcode in {target_format};")
        print(f'제거 대상 해시코드(재정제 대상) 제거 완료.')

        return

    def create_backup_table(self, db_name, tb_name):
        db_engine = self.db_engine

        backup_date = dt.datetime.now().strftime('%Y%m%d')
        backup_name = '_'.join([tb_name, backup_date])

        result = db_engine.execute(f"select count(*) from {db_name}.{tb_name};")
        result_cnt = result.first()[0]
        print(f'#'*30, f'현재 {tb_name} 총 데이터: {result_cnt}', '#'*30)

        db_engine.execute(f"create table if not exists {db_name}.{backup_name} select * from {db_name}.{tb_name};")
        print(f'현 MD 테이블 백업 완료.')

        return result_cnt
    
    def drop_table(self, db_name, tb_name):
        db_engine = self.db_engine
        db_engine.execute(f"drop table {db_name}.{tb_name};")

        print(f'{db_name}.{tb_name} 테이블 삭제 완료.')

        return

    def set_primary_key(self, db_name, tb_name, pk):
        db_engine = self.db_engine

        pk_format = '(' + str(pk)[1:-1] + ')'
        db_engine.execute(f"alter table {db_name}.{tb_name} add primary key {pk_format};")
        print(f'DB PK 지정 완료.')
        return
    
    def show_tables(self, db_name):
        db_engine = self.db_engine

        show_tables_sql = f'SHOW TABLES FROM {db_name}'
        tables = db_engine.execute(show_tables_sql)
        tables = tables.fetchall()

        if tables:
            tables = list(map(lambda x : x[0], tables))
        else:
            tables = []

        print(f'{db_name} 테이블 리스트 조회 완료.')
        return tables
    


class AutoDb(MysqlDb):
    def __init__(self, server_config):
        super().__init__(server_config)

    def get_latest_hist_data(self, db_name, tb_name, data_type):
        db_engine = self.db_engine

        sql = f"select * from {db_name}.{tb_name} where sync_purpose=\'{data_type}\' and isdrop=0 order by sync_run_date desc limit 1;"
        hist_data = pd.read_sql(sql=sql, con=db_engine, index_col=None)

        if len(hist_data) > 0:
            return {'sync_id': hist_data.loc[0, 'sync_id'], 'sync_purpose': hist_data.loc[0, 'sync_purpose'], 'sync_svc': hist_data.loc[0, 'sync_svc'], 'sync_run_date': hist_data.loc[0, 'sync_run_date'], 'sync_rtime': hist_data.loc[0, 'sync_rtime']}

        return {}
    
    def get_all_hashcodes(self, db_name, tb_name):
        db_engine = self.db_engine

        sql = f"select distinct(hashcode) from {db_name}.{tb_name};"
        distinct_hashcodes = pd.read_sql(sql=sql, con=db_engine, index_col=None)
        return list(distinct_hashcodes['hashcode'])

    def insert_row_manual_adding_hist(self, db_name, tb_name, sync_info):
        db_engine = self.db_engine

        db_engine.execute(f"insert into {db_name}.{tb_name}(sync_id, sync_purpose, sync_svc, sync_run_date, sync_rtime) values (\'{sync_info['sync_id']}\', \'{sync_info['sync_purpose']}\', \'{sync_info['sync_svc']}\', \'{sync_info['sync_run_date']}\', \'{sync_info['sync_rtime']}\');")
        print(f'auto db 히스토리 레코드 생성 완료.')
        return

    def update_row_manual_adding_hist(self, db_name, tb_name, sync_info):
        db_engine = self.db_engine

        end_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db_engine.execute(f"update {db_name}.{tb_name} set etime=\'{end_time}\', done=1 where sync_id=\'{sync_info['sync_id']}\' and sync_purpose=\'{sync_info['sync_purpose']}\' and sync_svc=\'{sync_info['sync_svc']}\' and sync_run_date=\'{sync_info['sync_run_date']}\' and sync_rtime=\'{sync_info['sync_rtime']}\';")
        print(f'auto db 히스토리 레코드 업데이트 완료.')
        return
    
    # 기간 만료된 백업 테이블 제거(2주 기준)
    def drop_expired_backup_tables(self, db_name, purpose, today):
        print('#' * 100, 'AUTO DB 기간 만료 백업 테이블 삭제', '#' * 100)

        tables = super().show_tables(db_name)
        drop_tables = list()

        if tables:
            today_date = parse(today)
            two_weeks_ago_date = today_date - dt.timedelta(days=14)
            # 기간 만료 확인(2주)
            two_weeks_ago_str = two_weeks_ago_date.strftime('%Y%m%d')
            for x in tables:
                if x.startswith(f'mass_final_md_data_{purpose}_') or x.startswith(f'mass_md_data_{purpose}_'):
                    backup_date = x.split('_')[-1]
                    if backup_date.isdigit() and two_weeks_ago_str > backup_date: # 2주보다 더 이전 백업 데이터일 경우 삭제
                        drop_tables.append(x)
                    else:
                        pass

        # DROP SQL문 생성 및 실행
        for tb_name in drop_tables:
            super().drop_table(db_name, tb_name)

        print('#' * 100, 'AUTO DB 기간 만료 백업 테이블 삭제 완료', '#' * 100)
        return
    
    def tmp_excute_query(self):
        db_engine = self.db_engine

        query = f"""select * from lg_cns_22cat.mass_final_md_data_po
                    where hashcode in (select distinct hashcode 
					from lg_cns_22cat.mass_md_data_po 
                    where reg_date = '');"""
        
        target_df = pd.read_sql(sql=query, con=db_engine, index_col=None)
        
        return target_df

    def tmp_update_rows(self, df):
        db_engine = self.db_engine

        for row in tqdm(df.itertuples()):
            db_engine.execute(f"update lg_cns_22cat.mass_md_data_po set reg_date=\'{row.reg_date}\' where hashcode=\'{row.hashcode}\';")
        print(f'auto db 히스토리 레코드 업데이트 완료.')
        return
    


class BonDb(MysqlDb):
    def __init__(self, server_config):
        super().__init__(server_config)
    
    ####################################################################################################################
    ### ○ 재정제 관련 함수 오버라이딩(오버로딩)
    ####################################################################################################################
    '''
    ○ 재정제 관련 함수 오버라이딩(오버로딩)
    '''
    # IMPORT
    def import_manual_data(self, db_name, tb_name, fix_match_date, fix_match=1):
        db_engine = self.db_engine

        sql = f"select * from {db_name}.{tb_name} where fix_match=\'{fix_match}\' and fix_match_date>=\'{fix_match_date}\';"   #  and fix_match_date>=\'{fix_match_date}\'
        print('실행 명령문 :', sql)
        data_iter = pd.read_sql(sql=sql, con=db_engine, index_col=None, chunksize=100000)
        print(f'데이터 iterator 생성완료 : {data_iter}')
        data = pd.DataFrame()
        for i, iter in enumerate(data_iter): # 10000 건씩 iterator
            print(i, iter)
            # if i > 10:
            #     break
            data = pd.concat([data, iter])

        for col in data.columns:
            data = data.astype({col : 'str'})

        print('데이터 가져오기 완료 :', data.head())
        return data

    def check_manual_work_done(self, db_name, tb_name, data_type, check_date):
        db_engine = self.db_engine

        sql = f"select * from {db_name}.{tb_name} where id=\'lgcns_mi_ecsm\' and purpose=\'{data_type}\' and svc=\'manual_md\' and run_date=\'{check_date}\' and done=1;"
        print('실행 명령문 :', sql)
        status_db = pd.read_sql(sql=sql, con=db_engine, index_col=None)

        print('#'*50)
        print(status_db)
        print('#'*50)
        return status_db
    
    def insert_row_sync_master(self, db_name, tb_name, sync_info):
        db_engine = self.db_engine

        start_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db_engine.execute(f"insert into {db_name}.{tb_name}(id, purpose, svc, stime, run_date) values (\'{sync_info['sync_id']}\', \'{sync_info['sync_purpose']}\', \'manual_md_on\', \'{start_time}\', \'{sync_info['sync_run_date']}\');")
        print(f'bon db 히스토리 레코드 생성 완료.')
        return start_time

    def update_row_sync_master(self, db_name, tb_name, sync_info):
        db_engine = self.db_engine

        end_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db_engine.execute(f"update {db_name}.{tb_name} set etime=\'{end_time}\', done=1 where id=\'{sync_info['sync_id']}\' and purpose=\'{sync_info['sync_purpose']}\' and svc=\'manual_md_on\' and run_date=\'{sync_info['sync_run_date']}\';")
        print(f'bon db 히스토리 레코드 업데이트 완료.')
        return end_time
    


class ESDb():
    def __init__(self, server_config, purpose):
        '''
        ○ 서버 및 DB 접근 관련 세팅
        '''
        # 서버 접속 정보
        self.server_config = server_config

        site_url = server_config['url'] + '/'
        if purpose == 'po':
            site_url += server_config['po_index'] + '/_search'
        elif purpose == 'ro':
            site_url += server_config['ro_index'] + '/_search'
        elif purpose == 'so':
            site_url += server_config['so_index'] + '/_search'

        self.site_url = site_url
        
        std_category_dict = {
            # 식기세척기
            '식기세척기': '식기세척기',
            # TV
            'TV' : 'TV',
            # 세탁기+건조기+의류관리기
            '세탁기+건조기+의류관리기' : '세탁기+건조기+의류관리기',
            '세탁기' : '세탁기+건조기+의류관리기',
            '건조기' : '세탁기+건조기+의류관리기',
            '세탁기+건조기' : '세탁기+건조기+의류관리기',
            '의류관리기(스타일러)' : '세탁기+건조기+의류관리기',
            '워시타워' : '세탁기+건조기+의류관리기',
            # 냉장고
            '김치냉장고' : '냉장고',
            '냉장고' : '냉장고',
            # 노트북
            '노트북' : '노트북',
            # 청소기
            '청소기' : '청소기',
            '로봇청소기' : '청소기',
            '무선청소기' : '청소기',
            # 모니터
            '모니터': '모니터',
            # 전기레인지
            '전기레인지': '전기레인지',
            # 에어컨
            '에어컨': '에어컨',
            # 공기청정기
            '공기청정기': '공기청정기',
            # 제습기
            '제습기': '제습기',
            # 전체
            '가전전체': '전체',
            '가전전체딜': '전체',
        }

        self.std_category_dict = std_category_dict

    def import_data(self, target_hashcodes=[]):
        SPLIT_CNT = self.server_config['split_size']

        query = {
            "size": SPLIT_CNT,
            "query" : {"match_all" : {}},
            "sort" : [
                {"hashcode" : "asc"} # 정렬기준
            ]
        }

        if target_hashcodes:
            query = {
                "size": SPLIT_CNT,
                "query" : {
                    "terms": {"hashcode": target_hashcodes}
                },
                "sort" : [
                    {"hashcode" : "asc"} # 정렬기준
                ]
            } 

        # Elastic Search에서 불러오기 전 세팅
        col_list = ['hashcode','title','option_name','mall_pid','bon_malls','bon_hash_org_rtime', 'bon_hash_rtime', 'bon_category_groups']
        all_df = pd.DataFrame(columns=col_list)  # Elastic Search에서 불러온 데이터를 담을 데이터 프레임 선언
        all_cnt = 0  # Elastic Search에서 불러온 데이터 건수

        # HASH INDEX FIELD 값 정의 함수
        def field_selection(x, col):
            if col in x['_source'].keys():
                fd_value = x['_source'][col]
                if (col=='bon_malls'):
                    if (type(fd_value) == list):
                        fd_value = fd_value[0]
                    else: # 그대로
                        pass
            # 'mall' 정보가 리스트 형태 또는 string
            elif col == 'bon_malls':
                fd_value = x['_source']['mall']
                if (type(fd_value) == list):
                    fd_value = fd_value[0]
                else: # 그대로
                    pass
            elif col == 'bon_category_groups':
                if 'keep' in x['_source'].keys():
                    fd_value = x['_source']['keep']['content_type']
                else:
                    fd_value = ''
            else:
                fd_value = ''

            return fd_value

        try:
            # Elastic Search에서 POST 방식으로 값 가져오기
            get_url = requests.post(url=self.site_url, headers=self.server_config['header'], data=json.dumps(query))
            es_json = get_url.json()

            # 데이터 리스트 추출
            data_value_list = list(map(lambda x: [field_selection(x, col) for col in col_list], es_json['hits']['hits']))

            # 2000건 데이터 프레임 생성
            df = pd.DataFrame(columns=col_list, data=data_value_list)

            # 시간 단위 계산
            df.loc[:, 'bon_hash_rtime'] = df['bon_hash_rtime'].map(lambda x : dt.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d') if x != '' else '')
            df.loc[:, 'bon_hash_org_rtime'] = df['bon_hash_org_rtime'].map(lambda x : dt.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d') if x != '' else '')

            # 전체 데이터프레임에 적재
            all_df = pd.concat([all_df, df])

            # 전체 건수 조회
            all_cnt = es_json['hits']['total'] # 11/27 일까지 전체 건 3,882,638건
            # 1317647(bon_date : 10월 20일 이전(20일까지 포함) 기준 건수) 2022-08-04 최소

            # for문으로 나머지 건들 iterator로 데이터 추출 및 적재
            for_count = int(all_cnt//SPLIT_CNT + np.ceil((all_cnt % SPLIT_CNT / SPLIT_CNT)))-1 # 추가 적업해야할 건수
            if for_count != 0:
                for i in range(0, for_count):
                    print('#' * 40 + ' 인덱스', i, '번째 for문 작업 시작', '#' * 40)
                    after_search_value_list = es_json['hits']['hits'][SPLIT_CNT - 1]['sort']  # 직전 작업의 마지막 인덱스의 해시코드
                    query['search_after'] = after_search_value_list  # Search_After 설정
                    # URL로 데이터 가져오기
                    get_url = requests.post(url=self.site_url, headers=self.server_config['header'], data=json.dumps(query))
                    es_json = get_url.json()  # Elastic Serach에서 새로운 건 검색/추출

                    # 데이터 리스트 추출
                    data_value_list = list(map(lambda x: [field_selection(x, col) for col in col_list], es_json['hits']['hits']))

                    # 10000건 데이터 프레임 생성
                    df = pd.DataFrame(columns=col_list, data=data_value_list)

                    # bon_rtime => 밀리 초 단위를 날짜로 변환
                    df.loc[:, 'bon_hash_rtime'] = df['bon_hash_rtime'].map(lambda x: dt.datetime.fromtimestamp(x / 1000).strftime('%Y-%m-%d') if x != '' else '')
                    df.loc[:, 'bon_hash_org_rtime'] = df['bon_hash_org_rtime'].map(lambda x : dt.datetime.fromtimestamp(x/1000).strftime('%Y-%m-%d') if x != '' else '')

                    # 전체 데이터프레임에 적재
                    all_df = pd.concat([all_df, df])
                    print('적재완료!! || 전체 적재 건수 :', len(all_df), '|| 이번 작업의 적재된 건수', len(df))
                    print('#' * 100)
            else:
                print('적재완료!! || 전체 적재 건수 :', len(all_df), '|| 이번 작업의 적재된 건수', len(df))
                print('#' * 100)
                pass
        except:
            print('에러 발생!!!', traceback.format_exc())

        # 전체 건수 수집 확인
        if all_cnt == len(all_df):
            print('전체건수 수집완료!!', len(all_df['hashcode'].unique()))
        else:
            print('ELASTIC SEARCH 전체건수 :', all_cnt, '|| 수집한 전체건수 :', len(all_df))

        # search_keyword 정의 함수
        def define_search_keyword(x):
            # join 한 'search_keyword' 결과에 따른 정의
            if (x in ['', '해당카테고리부재', '가전전체딜']):
                search_keyword = '전체'
            else:
                search_keyword = x

            return search_keyword

        # 서버 MYSQL 해시데이터에 INSERT
        if len(all_df) > 0:
            # search_keyword 그룹핑 정제 카테고리 표준화 작업
            # bon_category_groups에 리스트에 원소 가져오기
            # 그룹핑 카테고리 표준화 딕셔너리 => 표준화 카테고리로 치환
            # 딕셔너리에 원소에 대한 MAPPING 정보가 모두 들어있어야함
            # insert_df.loc[:, 'new_search_keyword'] = ''
            std_category_dict = self.std_category_dict
            
            std_cat_list = list()
            for i in tqdm(all_df.index):
                bon_c_grp = all_df.loc[i, 'bon_category_groups']
                std_cat = '해당카테고리부재'

                if type(bon_c_grp) == str:
                    if bon_c_grp in std_category_dict.keys():
                        std_cat = std_category_dict[bon_c_grp]
                    else:
                        std_cat = '전체'
                else:
                    if bon_c_grp:
                        for cat in bon_c_grp:
                            if cat in std_category_dict.keys():
                                std_cat = std_category_dict[cat]
                                break
                            else:
                                std_cat = '전체'
                    else: # 'bon_category_groups' 정보가 없을 경우
                        std_cat = '전체'

                # 변환 함수 적용
                std_cat = define_search_keyword(std_cat)
                # 리스트에 적재
                std_cat_list.append(std_cat)

        # 그룹핑 카테고리 값 치환
        all_df.loc[:, 'search_keyword'] = std_cat_list
        # 'bon_category_groups' : 리스트 => string으로 형태변환
        all_df = all_df.astype({'bon_category_groups' : 'str'})
        for idx, row in enumerate(all_df.itertuples()):
            if row.bon_hash_org_rtime != '':
                all_df.loc[idx, 'reg_date'] = parse(row.bon_hash_org_rtime).strftime('%Y-%m-%d')
            elif row.bon_hash_rtime != '':
                all_df.loc[idx, 'reg_date'] = (parse(row.bon_hash_rtime) - dt.timedelta(1)).strftime('%Y-%m-%d')   # 수집 날짜 컬럼 생성
 
        all_df.loc[:, 'reg_time'] = dt.datetime.today().strftime('%Y-%m-%d %H:%M:%S')  # 등록 날짜 컬럼 생성
        
        all_df = all_df[[col for col in all_df.columns if col != 'bon_hash_org_rtime']]
        all_df = all_df.sort_values(by='reg_date', ascending=False)
        all_df = all_df.drop_duplicates(subset='hashcode').reset_index(drop=True)
        return all_df