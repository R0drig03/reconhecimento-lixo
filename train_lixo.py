from ultralytics import YOLO

#para marcar as imagens
#https://www.makesense.ai/
def main():
    
    model = YOLO("yolov8n.pt")  #Carregando modelo do yolo

    print('INICIANDO O TREINAMENTO')
    model.train(data="lixo.yaml", epochs=30)  #Realizando o treinamento
    metrics = model.val() 
    
    print('TREINAMENTO FINALIZADO')


if __name__ == '__main__':
    main()