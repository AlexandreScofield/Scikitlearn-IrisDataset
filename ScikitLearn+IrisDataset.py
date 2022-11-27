from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
model = LinearSVC(random_state = 0, max_iter=5000)
iris = load_iris()
X = iris.data
y = iris.target
model.fit(X,y)
a = [5.1, 3.5, 1.4, 0.2]
b = [5.6, 3. , 4.1, 1.3]
c = [5.9, 3. , 5.1, 1.8]
testes = [a,b,c]
print(model.predict([a]))
previsões = model.predict(testes)
testes_classes = [0,1,2]
taxa_de_acerto = accuracy_score(testes_classes, previsões)
print("Taxa de acerto", taxa_de_acerto * 100)
    
