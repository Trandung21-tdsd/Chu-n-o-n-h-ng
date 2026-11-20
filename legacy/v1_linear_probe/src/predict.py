import argparse,json
from pathlib import Path
import torch
from torch import nn
from PIL import Image
from .model import backbone,preprocess

NAMES={'Carcinoma_in_situ':'Carcinoma tại chỗ','Cyst':'Nang dây thanh','High_grade_dysplasia':'Loạn sản độ cao','Hyperkeratosis':'Tăng sừng','Low_grade_dysplasia':'Loạn sản độ thấp','Other_benign':'Nhóm tổn thương lành tính khác','Papilloma':'U nhú','Polyp':'Polyp dây thanh','Reinkes_edema':'Phù Reinke','SCC':'Ung thư biểu mô tế bào vảy (nhãn dự đoán)'}

class Predictor:
    def __init__(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True);self.ck=ck
        self.encoder,dim=backbone(ck['config']['model'],False);self.encoder.load_state_dict(ck['encoder']);self.encoder.eval()
        self.head=nn.Linear(dim,len(ck['classes']));self.head.load_state_dict(ck['head']);self.head.eval()
    def predict(self,image,modality,threshold=.6):
        if modality!='CE-NBI':return {'status':'out_of_scope','message':'Model chỉ được huấn luyện trên CE-NBI thanh quản; chưa hỗ trợ ảnh họng thường, amidan hoặc nội soi ánh sáng trắng.'}
        if min(image.size)<64:return {'status':'insufficient_quality','message':'Ảnh quá nhỏ để phân tích.'}
        if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
        with torch.inference_mode():
            x=preprocess(image,self.ck['config']['clahe']).unsqueeze(0);z=self.encoder(x)
            p=self.head((z-self.ck['mean'])/self.ck['std']).softmax(1)[0].tolist()
        order=sorted(range(len(p)),key=lambda i:p[i],reverse=True)
        return {'status':'research_prediction' if p[order[0]]>=threshold else 'uncertain','scope':'CE-NBI laryngeal lesion classification; no healthy control class','run_id':self.ck['config']['run_id'],'top3':[{'label':self.ck['classes'][i],'name_vi':NAMES.get(self.ck['classes'][i],self.ck['classes'][i]),'score':p[i]} for i in order[:3]],'probabilities':dict(zip(self.ck['classes'],p)),'threshold':threshold,'note':'Điểm softmax chưa hiệu chuẩn, không phải xác suất mắc bệnh. Ngưỡng từ chối chỉ là heuristic, không phát hiện được ảnh ngoài phân phối. Cần bác sĩ đánh giá; không xác nhận ung thư bằng ảnh này.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('image',type=Path);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--modality',required=True,choices=['CE-NBI','white-light','unknown']);ap.add_argument('--threshold',type=float,default=.6);a=ap.parse_args()
    path=a.checkpoint or Path(json.loads(Path('runs/best_model.json').read_text())['checkpoint'])
    with Image.open(a.image) as im:result=Predictor(path).predict(im,a.modality,a.threshold)
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
