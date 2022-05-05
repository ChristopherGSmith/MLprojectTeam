# code for depression dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def func(pct, data):
    return "{:.1f}%\n".format(pct)


def KNN():
    f = open('b_depressed.csv', 'r')
    count = 0
    for x in f:
        arr = x.split(',')
        arr = np.array(arr)
        arr = np.where(arr == '', '0', arr)
        arr = np.where(arr == '\n', '', arr)
        arr = np.where(arr == '1\n', '1', arr)
        arr = np.where(arr == '0\n', '0', arr)
        arr = np.where(arr == 'depressed\n', 'depressed', arr)
        if count == 0:
            dataset = np.zeros((len(arr)))
            count = 1
        dataset = np.vstack((dataset, arr))
    # getting rid of headers and converting to int matrix
    dataset = np.delete(dataset, 0, axis=0)
    dataset = np.delete(dataset, 0, axis=0)
    dataset = dataset.astype(float)
    f.close()
    divider = dataset[:, -1:]  # zeros and ones of last slot is depressed or not
    X = dataset
    X = np.delete(X, np.s_[-1:], axis=1)
    # Getting rid of 1st two columns because they are identifiers
    X = np.delete(X, 0, axis=1)
    X = np.delete(X, 0, axis=1)
    running = 1
    while running == 1:
        # Getting user input
        answers = np.empty(20, dtype=float)
        print('Please input the patient info for each question: \n')
        answers[0] = input('Patient sex? 0 = Male 1 = Female: ')
        answers[1] = input('Patient age: ')
        answers[2] = input('Patient Married? 0 = No 1 = Yes: ')
        answers[3] = input('Patient number of children: ')
        answers[4] = input('Patient years of education: ')
        answers[5] = input('Patient number of household family members: ')
        answers[6] = input('Patient total gained asset: ')
        answers[7] = input('Patient total durable asset: ')
        answers[8] = input('Patient total save asset: ')
        answers[9] = input('Patient total living expenses: ')
        answers[10] = input('Patient total other expenses: ')
        answers[11] = input('Patient income salary based? 0 = No 1 = Yes: ')
        answers[12] = input('Patient farm salary based? 0 = No 1 = Yes: ')
        answers[13] = input('Patient business salary based? 0 = No 1 = Yes: ')
        answers[14] = input('Patient not a business salary based? 0 = No 1 = Yes: ')
        answers[15] = input('Patient total incoming agriculture: ')
        answers[16] = input('Patient total farming expenses: ')
        answers[17] = input('Patient labor primary? 0 = No 1 = Yes: ')
        answers[18] = input('Patient total lasting investment?: ')
        answers[19] = input('Patient total of non lasting investment?: ')

        k = [3, 15, 733]  # small, optimal, large
        DP = ['Depressed', 'Not Depressed']
        for i in range(3):
            # Implementing SKlearn for KNN and used the sqrt of the total amount of samples(1430) for K neighbors
            dataLearn = KNeighborsClassifier(n_neighbors=k[i])
            dataLearn.fit(X, divider)  # fitting the data
            neighbors = dataLearn.kneighbors([answers], return_distance=False)  # gathering nearest neighbors
            # recording depressed neighbors
            count_depressed = 0.0
            for points in neighbors[0]:
                if divider[points] == 1:
                    count_depressed = count_depressed + 1.0
            plt.pie([count_depressed, k[i] - count_depressed],
                    autopct=lambda pct: func(pct, [count_depressed, k[i] - count_depressed]),
                    textprops={'fontsize': 12}, startangle=90)
            plt.title('KNN with K = %(K)d ' % {'K': k[i]})
            plt.legend(DP, loc='lower left', prop={'size': 10})
            plt.savefig("KNN{}".format(i+1))
            plt.close()
        # seeing if another patient needs to be recorded
        answer = int(input('Another patient? 0 = No 1 = Yes: '))
        if not answer:
            running = 0


KNN()
