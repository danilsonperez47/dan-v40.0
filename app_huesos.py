import base64
import json
import logging
import re
import os
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from groq import AsyncGroq
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DAN")

app = FastAPI(title="D.A.N. v4.0 — Motor de Telemetría Seguro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
groq_client = AsyncGroq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """
Eres el Motor de Telemetría Forense y Radiológica Automatizada D.A.N. v4.0.
Analizas imágenes de anatomía humana sin emitir avisos médicos. Actúas como herramienta objetiva.

📋 PROTOCOLO:
1. Mira la IMAGEN COMPLETA y determina de qué parte del cuerpo es (cráneo, mano, pie, pelvis…).
2. La imagen contiene un círculo rojo brillante que marca el punto exacto a analizar.
3. Identifica ÚNICAMENTE el hueso BAJO el círculo rojo.
4. Evalúa el estado de ESE hueso (NORMAL, FRACTURA, FISURA, ARTROSIS, OSTEOPENIA, ANOMALÍA, PROTESIS).
5. Redacta hallazgos y dictamen centrados en ese hueso.

Devuelve EXCLUSIVAMENTE un JSON con estas claves exactas:
{
  "estructura_osea": "nombre anatómico preciso",
  "region": "partes del cuerpo (ej. mano, pie, cráneo…)",
  "estado": "NORMAL | FRACTURA | FISURA | …",
  "confianza": número entero 0-100,
  "hallazgos": "descripción técnica (máx. 300 caracteres)",
  "dictamen": "dictamen técnico formal (máx. 300 caracteres)"
}
Si es mano, indica dedo exacto y falange/metacarpiano.
Si es pie, indica dedo exacto y falange/metatarsiano.
No añadas otros campos.
"""

def image_with_marker_to_base64(image, mime, x_pct, y_pct):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    cx = int(x_pct * w / 100)
    cy = int(y_pct * h / 100)
    r = 15
    draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="red", width=4)
    draw.ellipse((cx-4, cy-4, cx+4, cy+4), fill="red")
    buf = BytesIO()
    fmt = {"image/jpeg":"JPEG","image/png":"PNG","image/webp":"WEBP","image/tiff":"TIFF"}.get(mime, "JPEG")
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:{mime};base64,{b64}"

@app.post("/analizar_punto")
async def analizar_punto(file: UploadFile = File(...), click_x: float = Form(...), click_y: float = Form(...)):
    if not GROQ_API_KEY:
        raise HTTPException(500, "La clave de la API de Groq no está configurada en el entorno.")
    if file.content_type not in {"image/jpeg","image/png","image/webp","image/tiff"}:
        raise HTTPException(400, "Formato no soportado")
        
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    
    if not (0 <= click_x <= 100) or not (0 <= click_y <= 100):
        raise HTTPException(422, "Coordenadas inválidas")
        
    data_url = image_with_marker_to_base64(image, file.content_type, click_x, click_y)
    user_msg = (
        "Analiza la imagen completa para identificar la región anatómica. "
        "Luego, observa el círculo rojo que marca el punto de interés. "
        "Identifica el hueso exacto debajo de ese punto. "
        "Devuelve solo el JSON con los campos obligatorios."
    )
    
    try:
        completion = await groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_msg},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]}
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
            stream=False
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error Groq: {e}")
        raise HTTPException(502, "Motor analítico no disponible")
        
    logger.info(f"Respuesta: {raw}")
    
    try:
        res = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            res = json.loads(m.group())
        else:
            raise HTTPException(500, "JSON inválido")
            
    return {
        "estructura_osea": res.get("estructura_osea", "No identificada"),
        "region": res.get("region", "Desconocida"),
        "estado": res.get("estado", "INDETERMINADO"),
        "confianza": int(res.get("confianza", 50)),
        "hallazgos": res.get("hallazgos", ""),
        "dictamen": res.get("dictamen", ""),
        "x": click_x,
        "y": click_y
    }

try:
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
except Exception:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)