import torch
from torch.utils.data import Dataset # iterator 역할


class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples): # DataLoader에서 설정된 batch-size 만큼의 행이 iterator 하게 들어옴
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]
        print('samples 길이 확인 :', len(texts), len(labels))
        encoding = self.tokenizer(
            texts,
            padding=True, # batch내에서 가장 긴 길이에 맞춰서 padding 한다.
            truncation=True, # 모델이 입력으로 받을 수 있는 최대의 길이에 맞춰서 truncation 한다.(토큰 최대길이)
            return_tensors="pt", # torch.Tensor : pt, tf.constant : tf, np.ndarray로 결과를 리턴한다.
            max_length=self.max_length # 모델이 입력으로 받을 최대의 입력 길이를 설정한다.
        )
        # print('tokenzier 토큰화 및 인코딩 후 :', encoding)
        return_value = {
            'input_ids': encoding['input_ids'], # encoding 결과
            'attention_mask': encoding['attention_mask'], # 학습에 집중할 attention_mask
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        # print('인코딩 input_ids :',return_value['input_ids'])
        # print('인코딩 attention_mask :',return_value['attention_mask'])
        # print('인코딩 labels :',return_value['labels'])

        if self.with_text:
            return_value['text'] = texts
        # print('인코딩 text :', return_value['labels'])
        return return_value


class TextClassificationDataset(Dataset): # 클래스(상속할 클래스) => 상속할 클래스의 함수는 다 가지고 있다. 거기에다가 새롭게 override 해준거

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self): # len(해당 객체) 시에 행의 길이를 Return 한다
        return len(self.texts)
    
    def __getitem__(self, item): # 해당 객체[인덱스] 원하는 인덱스에 접근할 수 있고 그에 해당하는 값을 Return 한다
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        } # 딕셔너리 형태로 리턴
