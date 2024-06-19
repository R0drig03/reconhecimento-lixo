from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

image_path = 'D:\\PythonProjects\\projeto-cv2\\intervenção_praça_migrantes\\garrafa1.jpg'

# Carrega a imagem
img = cv2.imread(image_path)

# Verifica se a imagem foi carregada corretamente
if img is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit()

# Carrega o modelo YOLO treinado
model = YOLO(r"runs\\detect\\train6\\weights\\best.pt")

# Dicionário para armazenar o histórico de rastreamento dos objetos
track_history = defaultdict(lambda: [])
# Variáveis para controle de rastreamento e rastros
seguir = False
deixar_rastro = False

# Define o limite de confiança (70%)
confidence_threshold = 0.1

if seguir:
    results = model.track(img, persist=True)
else:
    results = model(img)

# Processa a lista de resultados
for result in results:
    # Obtém as caixas delimitadoras e confidências
    boxes = result.boxes
    if boxes is not None:
        # Filtra as detecções com base no limite de confiança
        for box in boxes:
            if box.conf >= confidence_threshold:
                
                # Desenha o retângulo na imagem
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Adiciona o nome do objeto identificado
                label = model.names[int(box.cls)]  # Nome da classe
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Se seguir e deixar rastro estiverem ativos, desenha as trilhas
                if seguir and deixar_rastro:
                    try:
                        # Obtém o ID de rastreamento
                        track_id = int(box.id)
                        x, y = (x1 + x2) / 2, (y1 + y2) / 2
                        track = track_history[track_id]
                        track.append((x, y))  # ponto central (x, y)
                        if len(track) > 30:  # mantém 30 rastros para 30 frames
                            track.pop(0)

                        # Desenha as linhas de rastreamento
                        points = np.array(track, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
                    except:
                        pass

# Obtém as dimensões da tela
screen_width = cv2.getWindowImageRect("Tela")[2]
screen_height = cv2.getWindowImageRect("Tela")[3]

# Redimensiona a imagem mantendo a proporção
h, w = img.shape[:2]
scale = min(screen_width / w, screen_height / h)
new_w, new_h = int(w * scale), int(h * scale)
resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

# Configura a janela do OpenCV para tela cheia
cv2.namedWindow("Tela", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Tela", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Mostra a imagem processada em uma janela
cv2.imshow("Tela", resized_img)

# Espera por uma tecla 'q' para sair
while True:
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Fecha todas as janelas abertas
cv2.destroyAllWindows()
print("desligando")
