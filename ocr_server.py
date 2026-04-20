from fastapi import FastAPI, Form,HTTPException
from pydantic import BaseModel
from typing import Optional
import cv2
import re
import os
import numpy as np
import logging
import json
import base64
from paddleocr import PaddleOCR

# 屏蔽无用日志
logging.getLogger("ppocr").setLevel(logging.ERROR)
# --- 关键：在 import paddle 之前通过环境变量禁用 oneDNN ---
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_enable_pir_api'] = '0'
# 禁用模型源检查，加快启动速度
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

app = FastAPI(title="Stock OCR API")

# --- 全局初始化 OCR ---
# 放在外面确保模型只加载一次，常驻内存，提升响应速度
# 修改这一行
ocr = PaddleOCR(
    lang='ch', 
    device='cpu', 
    use_textline_orientation=True  # 替代旧的 use_angle_cls
)
# 定义请求参数模型
class OCRRequest(BaseModel):
    image_path: str

# --- 原有的解析逻辑保持不变，但封装成内部函数 ---

def group_by_line(result):
    all_elements = []
    for block in result:
        if 'rec_texts' not in block: continue
        texts = block['rec_texts']
        boxes = block['rec_boxes']
        for text, box in zip(texts, boxes):
            box = np.array(box).reshape(-1, 2)
            y = box[:, 1].mean()
            x = box[:, 0].mean()
            all_elements.append({'y': y, 'x': x, 'text': text})

    if not all_elements: return []
    all_elements.sort(key=lambda e: e['y'])

    grouped = []
    current_line = [all_elements[0]]
    last_y = all_elements[0]['y']

    for i in range(1, len(all_elements)):
        e = all_elements[i]
        if abs(e['y'] - last_y) < 15:
            current_line.append(e)
        else:
            current_line.sort(key=lambda e: e['x'])
            grouped.append([item['text'] for item in current_line])
            current_line = [e]
        last_y = e['y']

    if current_line:
        current_line.sort(key=lambda e: e['x'])
        grouped.append([item['text'] for item in current_line])
    return grouped

def parse_funds(lines):
    funds = []
    current_fund_name = ""
    for line in lines:
        line_str_for_check = "".join(line).replace(" ", "")
        if re.search(r'[\u4e00-\u9fa5]', line_str_for_check):
            exclude = ['收益', '排序', '资产', '全部', '占比', '金额', 'A股']
            if not any(k in line_str_for_check for k in exclude):
                current_fund_name = line_str_for_check
            continue

        joined_line = " ".join(line)
        joined_line = re.sub(r"(?<=\d)([-+])", r" \1", joined_line)
        nums_raw = re.findall(r"[-+]?\d[\d,]*\.?\d*", joined_line)
        nums = []
        for n in nums_raw:
            try:
                clean_n = n.replace(",", "")
                if clean_n.count('.') > 1:
                    parts = clean_n.split('.')
                    clean_n = "".join(parts[:-1]) + "." + parts[-1]
                nums.append(float(clean_n))
            except: continue

        if len(nums) >= 3 and current_fund_name:
            funds.append({
                "name": current_fund_name,
                "amount": nums[0],
                "total": nums[2]
            })
            current_fund_name = ""
    return funds

# --- API 路由定义 ---

@app.post("/predict")
async def predict(
    image_path: Optional[str] = Form(None), 
    base64_str: Optional[str] = Form(None)
):
    """
    既支持 image_path (本地路径)，也支持 base64_str (图片Base64编码)
    """
    img = None

    # 1. 优先尝试处理 Base64
    if base64_str:
        try:
            # 去掉可能存在的 base64 头 (如 data:image/jpeg;base64,)
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            
            img_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Base64 解码后的数据无法识别为图片")
            print("🚀 成功解析 Base64 图片")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 解析失败: {str(e)}")

    # 2. 如果没有 Base64，尝试处理文件路径
    elif image_path:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"找不到文件: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise HTTPException(status_code=400, detail="图片路径读取失败或格式错误")
        print(f"📂 成功读取本地图片: {image_path}")

    # 3. 如果两个都没传
    else:
        raise HTTPException(status_code=400, detail="请提供 image_path 或 base64_str 参数")

    # --- 后续 OCR 逻辑保持不变 ---
    try:
        result = ocr.predict(img)
        lines = group_by_line(result)
        funds_data = parse_funds(lines)
        
        return {
            "success": True,
            "data": funds_data,
            "raw_lines": lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别过程出错: {str(e)}")

  

@app.get("/health")
async def health():
    return {"status": "ok2"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8002)