import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. Импортируем подготовку данных и модель из ваших файлов
from vectorize import get_prepared_data
from model import TextClassifier

def main():
    # Настройка процессора/видеокарты
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Загружаем тензоры
    X, y = get_prepared_data()
    num_classes = len(torch.unique(y))
    in_features = X.shape[1] # Берем количество признаков (1000)
    
    # 3. Делим выборку на Train (80%) и Test (20%)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. Создаем модель
    model = TextClassifier(in_features=in_features, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Цикл обучения
    epochs = 10
    print(f"Старт обучения на {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        # Валидация
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        print(f"Эпоха [{epoch+1}/{epochs}] | Loss: {running_loss/len(train_dataset):.4f} | Accuracy: {correct/total:.4f}")

# КРИТИЧЕСКИ ВАЖНО: Проверьте наличие этих двух строк в самом низу train.py!
if __name__ == "__main__":
    main()
