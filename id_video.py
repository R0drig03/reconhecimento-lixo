from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

video_path = 'D:\PythonProjects\projeto-cv2\intervenção_praça_migrantes\essa.jpg'

cap = cv2.VideoCapture(video_path)

model = YOLO(r"runs\\detect\\train6\\weights\\best.pt")

track_history = defaultdict(lambda: [])

seguir = False
deixar_rastro = False

confidence_threshold = 0.7

while True:
    success, img = cap.read()  # Captura um frame do vídeo

    if not success:
        break  # Sai do loop se não conseguir ler um frame (fim do vídeo)

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
                    # Desenha o retângulo no frame
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

    cv2.imshow("Tela", img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("desligando")
