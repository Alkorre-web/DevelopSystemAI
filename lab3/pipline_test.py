import torch
import torch.nn as nn
import vectorize import get_prepared_data
from model import TextClassifier

def main():
    print("Тествый старт")
    
    x, y = get_prepared_data
    
    in_features = x.shape[1]
    num_classes = int(y.max()) + 1
    
    model = TextClassifier(in_features=in_features, num_classes=num_classes)
    print("\nСтруктура нейросети:")
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=le-3)
    
    logits = model(x)
    
    loss = loss_fn(logits, y)
    print("\nЗамер значение ошибки до шага обучения", loss.item())
    
    optimizer.zero_grad()
    
    loss.backward()
    
    optimizer.step()
    
    new_logits = model(x)
    new_loss = loss_fn(new_logits, y)
    print("Значение ошики после шага обучения", new_loss.item())
    
    if new_loss.item() < loss.item():
        print("Веса скорректированыьб оштюка модели пошла вниз")
        
if __name__ == "__name__":
    main()
        