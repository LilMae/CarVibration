import os
import pandas as pd
import shutil
import argparse

CLASS_LIST = ['Aggressive_breaking', 'Aggressive_acceleration', 'Aggressive_left_turn', 'Aggressive_right_turn',
            'Aggressive_left_lane_change', 'Aggressive_right_lane_change', 'Non_aggressive_event']

def load_data_sheet(csv_path):
    """Label 정보는 초(second)단위로 구성되어 있는데,
    데이터 정보가 나노초(secondNano)단위로 구성되어 있어서,
    second를 기준으로 맞추어 준다.

    Args:
        csv_path (str): 데이터 csv 파일의 경로

    Returns:
        pd.DataFrame: csv데이터를 읽고 초단위로 정렬된 결과
    """
    data_pd = pd.read_csv(csv_path)
    
    data_pd['uptime'] = (data_pd['uptimeNanos']/1e9).round(2)
    start_time = data_pd['uptime'][0]
    data_pd['uptime'] = data_pd['uptime'] - start_time
    data_pd = data_pd.drop(['uptimeNanos', 'timestamp'], axis=1)
    
    return data_pd

def data_slice(data_pd, gt_pd):
    """원본 csv는 시간 레이블 단위로 데이터가 잘려져 있지 않음
    gt 정보를 이용해 데이터를 csv데이터를 분할하고 레이블로 정리해 주는 코드

    Args:
        data_pd (pd.DataFrame): load_data_sheet() 를 이용해 csv 파일을 읽은 데이터
        gt_pd (pd.DataFrame): groundTruth.csv 를 읽은 데이터

    Returns:
        dict : label 을 key로 가지고 해당 레이블에 맞게 슬라이싱 된 pd.DataFrame을 item으로 가지는 dict
    """
    
    # gt_pd에 데이터가 포루투칼어로 되어 있어 영어로 바꾸어주기 위한 dict
    nome_to_name = {
        'freada_agressiva' : 'Aggressive_breaking',
        'aceleracao_agressiva' : 'Aggressive_acceleration',
        'curva_esquerda_agressiva' : 'Aggressive_left_turn',
        'curva_direita_agressiva' : 'Aggressive_right_turn',
        'troca_faixa_esquerda_agressiva' : 'Aggressive_left_lane_change',
        'troca_faixa_direita_agressiva' : 'Aggressive_right_lane_change',
        'evento_nao_agressivo' : 'Non_aggressive_event'
    }
    

    # 슬라이싱된 pd.DataFrame이 클래스별로 저장될 dict
    data_dict = {'Aggressive_breaking' : [], 
                'Aggressive_acceleration' : [], 
                'Aggressive_left_turn' : [], 
                'Aggressive_right_turn' : [],
                'Aggressive_left_lane_change' : [], 
                'Aggressive_right_lane_change' : [], 
                'Non_aggressive_event' : []}

    for info in gt_pd.iterrows():
        
        info = info[1]   
        
        # 1. envento 에 해당하는 값이 classe_nome 가 된다.(영어로 변환)
        classe_nome = info['evento']
        class_name = nome_to_name[classe_nome]
        
        # 2. class의 시작 시간과 종료 시간을 가져온다.
        start_time = info[' inicio']
        end_time = info[' fim']

        # 3. 시작 시간과 종료시간을 기준으로 데이터를 잘라내고 data_dict에 저장해둔다.
        filted = data_pd[(data_pd['uptime'] >= start_time) & (data_pd['uptime']<end_time)]
        data_dict[class_name].append(filted)
    
    return data_dict

def data_convert(data_pd, gt_pd):
    """Change point detection을 위해 포인트별로 데이터를 구성하고 데이터의 변화양상을 분석할 수 있도록 하는 함수

    Args:
        data_pd (_type_): _description_
        gt_pd (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # 클래스를 인덱스로 저장하기 때문에 필요한 class list
    class_list = CLASS_LIST
    
    # gt_pd에 데이터가 포루투칼어로 되어 있어 영어로 바꾸어주기 위한 dict
    nome_to_name = {
        'freada_agressiva' : 'Aggressive_breaking',
        'aceleracao_agressiva' : 'Aggressive_acceleration',
        'curva_esquerda_agressiva' : 'Aggressive_left_turn',
        'curva_direita_agressiva' : 'Aggressive_right_turn',
        'troca_faixa_esquerda_agressiva' : 'Aggressive_left_lane_change',
        'troca_faixa_direita_agressiva' : 'Aggressive_right_lane_change',
        'evento_nao_agressivo' : 'Non_aggressive_event'
    }
    # 클래스 정보를 저장할 새로운 칼럼을 만들고 default 는 non-aggressive event 로 설정해 둔다.
    data_pd['class'] = 6

    for info in gt_pd.iterrows():
        
        info = info[1]   
        
        # 1. envento 에 해당하는 값이 classe_nome 가 된다.(영어로 변환)
        classe_nome = info['evento']
        class_name = nome_to_name[classe_nome]
        
        # 2. class의 시작 시간과 종료 시간을 가져온다.
        start_time = info[' inicio']
        end_time = info[' fim']

        # 3. 클래스 칼럼에 해당 
        data_pd.loc[(data_pd['uptime'] >= start_time) & (data_pd['uptime']<end_time), 'class'] = class_list.index(class_name)
        
    return data_pd
    
    
if __name__=='__main__':
    
    # Args Parsing
    parser = argparse.ArgumentParser(description = 'Data processing parse')
    parser.add_argument('-target_data', type=str,
                        help='one of acc, accline, mag, gy')
    args = parser.parse_args()
    
    # Data General Information : path, class information, car list
    data_root = os.path.join(os.getcwd(), 'data')
    class_list = CLASS_LIST
    car_list = ['16', '17', '20', '21']
    car_dict = {}
    
    """
    Step 1 - 데이터 읽어오기
    : Data를 폴더 구조에서 읽어오고 단위를 초(Sec)로 맞추어주는 작업
    """
    
    #차량별로 csv데이터 시트가 존재해서 각각 읽어 준다.
    for car_no in car_list:
        # 1. 차량을 번호로 저장해놓은 폴더 정보들
        car_folder = os.path.join(data_root, car_no)
        
        # 2. 각 폴더 내에 존재하는 데이터 파일(csv)의 경로 설정
        acc_path = os.path.join(car_folder, 'aceleracaoLinear_terra.csv')
        accline_path = os.path.join(car_folder, 'acelerometro_terra.csv')
        mag_path = os.path.join(car_folder, 'campoMagnetico_terra.csv')
        gyr_path = os.path.join(car_folder, 'giroscopio_terra.csv')
        
        # 3. 각 데이터 파일을 읽으면서 초(sec) 단위로 데이터를 조정해 줌
        acc, accline, mag, gyr = load_data_sheet(acc_path), \
            load_data_sheet(accline_path), load_data_sheet(mag_path), load_data_sheet(gyr_path)
        
        # 4. 클래스 정보가 담긴 파일(csv)를 읽기
        gt_path = os.path.join(car_folder, 'groundTruth.csv')
        gt = pd.read_csv(gt_path)

        # 5. 모든 데이터들은 dict 형태로 저장하여 사용할 수 있도록 함
        data_dict = {
            'acc' : acc,
            'accline' : accline,
            'mag' : mag,
            'gyr' : gyr,
            'gt' : gt
        }
        car_dict[car_no] = data_dict
        
    
    """
    Step 2-1 - 데이터 가공해서 저장하기 : classification
    : 읽어온 데이터 중 target_data에서 class를 나누고 class별로 따로 저장하는 부분
    """
    new_data_root =os.path.join(os.getcwd(), f'Trimmed_{args.target_data}')
    if os.path.isdir(new_data_root):
        shutil.rmtree(new_data_root)
    os.mkdir(new_data_root)

    for car_no in car_dict.keys():
        gt_pd = car_dict[car_no]['gt']
        data_pd = car_dict[car_no][args.target_data]
        
        sliced = data_slice(data_pd, gt_pd)
        for class_name in class_list:
            
            cls_folder = os.path.join(new_data_root, class_name)
            
            if not os.path.isdir(cls_folder):
                os.mkdir(cls_folder)
            
            for a_data in sliced[class_name]:
                idx = len(os.listdir(cls_folder))+1
                a_data.to_csv(os.path.join(cls_folder, f'{idx}.csv'))

    
    """
    Step 2-2 - 데이터 가공해서 저장하기 : change point detection
    : 읽어온 데이터 중 target_data에서 class를 자르지 않고, 옆에 클래스를 표기하여 저장하는 부분
    """
    new_data_root = os.path.join(os.getcwd(), f'Untrimmed_{args.target_data}')
    
    if os.path.isdir(new_data_root):
        shutil.rmtree(new_data_root)
    os.mkdir(new_data_root)

    for car_no in car_dict.keys():
        gt_pd = car_dict[car_no]['gt']
        data_pd = car_dict[car_no][args.target_data]
        
        data_preprocessed = data_convert(data_pd, gt_pd)
        
        data_preprocessed.to_csv(os.path.join(new_data_root, f'{car_no}.csv'))
        
    """
    Step 3 - 주어진 데이터의 클래스 정보를 담고 있는 txt파일 만들어 두기
    """
    with open('classes.txt', 'w') as f:
    
        for label in class_list:
            f.write(f'{label}\n')