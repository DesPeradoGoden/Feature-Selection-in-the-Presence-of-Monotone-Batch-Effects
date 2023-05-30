import numpy as np
import random
# numpy.random.seed(seed=993)
batch = ['1', '2', '3', '4']
shape = [[5, 10], [50, 10], [5, 100], [50, 100]]
# batch = ['4']
# shape = [[50, 1000]]
for batch_num in range(4):
    d = 100  # Dimension of data
    m = shape[batch_num][0]   # Number of datasets
    n = shape[batch_num][1]  # Number of data points in each dataset

    instances = m * n

    mean = list(np.zeros(d))   # use log-normal
    A = (np.random.rand(d)+10) / 5
    cov = np.diag(A)

    X = np.random.multivariate_normal(mean, cov, instances)

    theta = np.random.randn(d, 1)

    for i in range(len(theta)):
        t = random.randint(1, 10)
        if t > 1:
            theta[i][0] = 0

    theta = theta / np.linalg.norm(theta)

    noise = np.random.normal(0, 1, instances)   # non-Gaussian
    noise = noise[:, np.newaxis]

    Y = np.dot(X, theta)
    # print(np.linalg.norm(Y))
    full_data = np.concatenate((X, Y), axis=1)
    # print(full_data, full_data.shape)
    np_params = []

    for i in range(m):
        # Dataset i
        indi = np.random.randint(0, 2)
        # a_option = [np.random.lognormal(2.5, 1), np.random.lognormal(7, 0.5)]
        # b_option = [np.random.lognormal(2.5, 1), np.random.lognormal(7, 0.5)]
        # c_option = [np.random.lognormal(2.5, 1), np.random.lognormal(7, 0.5)]
        # a_i = a_option[indi]
        # b_i = b_option[indi]
        # c_i = c_option[indi]
        a_i = np.random.lognormal(2.5, 1)
        b_i = np.random.lognormal(2.5, 1)
        c_i = np.random.lognormal(2.5, 1)
        np_params.append([a_i, b_i, c_i])

        begin = int(i * instances / m)
        end = int((i+1) * instances / m)
        data_i = full_data[begin:end, :]
        real_data_i = full_data[begin:end, :].copy()
        name = 'Scenario'+batch[batch_num]+'/original_data_y_transformed' + str(i+1) + '.csv'
        np.savetxt(name, data_i, delimiter=",")

        for k in range(data_i.shape[1]):
            # batch_noise_mean = np.random.rand()*0
            # batch_noise_var = np.random.rand()*1000
            for j in range(data_i.shape[0]):
                temp = a_i * np.sign(data_i[j][k])*np.abs(data_i[j][k])**(5/3) + b_i * data_i[j][k] + c_i
                real_data_i[j][k] = temp
                # data_i[j][k] = a_i * data_i[j][k]**3 + b_i * data_i[j][k] + c_i + d_i*np.random.normal(0, 10)
                data_i[j][k] = a_i * np.sign(data_i[j][k])*np.abs(data_i[j][k])**(5/3) + b_i * data_i[j][k] + c_i + \
                               np.random.normal(0, 1)*np.sqrt(a_i)  # add Laplace noise
                # print(temp, data_i[j][k])

        name = 'Scenario'+batch[batch_num]+'/data_y_transformed' + str(i+1) + '.csv'
        name2 = 'Scenario' + batch[batch_num] + '/data_y_transformed(wo_noise)' + str(i + 1) + '.csv'
        np.savetxt(name, data_i, delimiter=",")
        np.savetxt(name2, real_data_i, delimiter=",")

    np.savetxt('Scenario'+batch[batch_num]+'/theta_y_transformed.csv', theta, delimiter=",")

    np.savetxt('Scenario'+batch[batch_num]+'/params_y_transformed.csv', np.array(np_params))

