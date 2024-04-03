import topsimp as ts
import numpy as np
import time
import matplotlib.pyplot as plt


""" img = np.array([[1,2,3], [4,5,0]])
uf = ts.UnionFindImg(img)
ti = time.time()
simplified_img = uf.compute(10)
tf = time.time()
persistence = uf.get_presistence()
print(f'The new image is \n {simplified_img} \n')
print(f'The original image\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')


img = np.array([[1,2,3], [4,5,0]])
uf = ts.UnionFindImg(img)
ti = time.time()
simplified_img = uf.compute(10)
tf = time.time()
persistence = uf.get_presistence()
print(f'The new image is \n {simplified_img} \n')
print(f'The original image\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')


img = np.array([[1,2,3], [4,5,0]])
uf = ts.UnionFindImg(img)
ti = time.time()
simplified_img = uf.compute(10)
tf = time.time()
persistence = uf.get_presistence()
print(f'The new image is \n {simplified_img} \n')
print(f'The original image\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')

img = np.array([[1,2,3], [4,5,0]])
uf = ts.UnionFindImg(img)
ti = time.time()
simplified_img = uf.compute(10)
tf = time.time()
persistence = uf.get_presistence()
print(f'The new image is \n {simplified_img} \n')
print(f'The original image\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')


F = [[1,2,3], [2,3,5,4]]
filtration = {1:0, 2:2, 3:3, 4:1, 5:1}



ti = time.time()
ufg = ts.UnionFindGraph()
ufg.from_faces(F, filtration)
persistence = ufg.compute(10)
tf = time.time()
simplified_graph = ufg.get_modified_filtration()
print(f'The new graph is \n {simplified_graph} \n')
print(f'The original graph\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')


ti = time.time()
ufg = ts.UnionFindGraph()
ufg.from_faces(F, filtration)
persistence = ufg.compute(10)
tf = time.time()
simplified_graph = ufg.get_modified_filtration()
print(f'The new graph is \n {simplified_graph} \n')
print(f'The original graph\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}') """


F = [[1,2,3], [2,3,5,4]]
filtration = {1:0, 2:2, 3:3, 4:1, 5:1}

ti = time.time()
ufg = ts.UnionFindGraph()
ufg.from_faces(F, filtration)
persistence = ufg.compute(10)
tf = time.time()
simplified_graph = ufg.get_modified_filtration()
print(f'The new graph is \n {simplified_graph} \n')
print(f'The original graph\'s persistence was:\n {persistence}')
print(f'Elapsed time: {tf-ti}')


ufg.draw(with_filtration=True, modified=True, with_labels=True)
plt.show()


