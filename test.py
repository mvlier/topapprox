import topsimp as ts
import numpy as np


img = np.array([[1,2,3], [4,5,0]])

uf = ts.UnionFindImg(img)

simplified_img = uf.compute(10)

persistence = uf.get_presistence()

print(f'The new image is \n {simplified_img} \n')
print(f'The original image\'s persistence was:\n {persistence}')




