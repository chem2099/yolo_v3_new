import os

with open('./data/car/train/train_1w.csv', 'w') as target:
    with open('./data/car/image/train_1w/train_1w.csv', 'r') as file:
        file.readline()

        for line in file:
            line = line.split(',')
            img_path = os.path.join('data', 'car', 'image', 'train_1w', line[0])

            line = img_path + ',' + line[1]
            target.write(line)

with open('./data/car/train/train_b.csv', 'w') as target:
    with open('./data/car/image/train_b/train_b.csv', 'r') as file:
        file.readline()

        for line in file:
            line = line.split(',')
            img_path = os.path.join('data', 'car', 'image', 'train_b', line[0])

            line = img_path + ',' + line[1]
            target.write(line)


# img_list = os.listdir('./data/car/image/test_a/')

# with open('./data/car/image/test_a/test_a.csv', 'w') as file:
#     file.write('name\n')

#     for name in img_list:
#         file.write(name + '\n')

# with open('./data/car/test/test_a.csv', 'w') as target:
#     with open('./data/car/image/test_a/test_a.csv', 'r') as file:
#         file.readline()

#         for line in file:
#             img_path = os.path.join('data', 'car', 'image', 'test_a', line)

#             target.write(img_path)




