import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import csv

# _listはリスト
# np_はnp.arrayに格納されている
# mat_はnp.matrixに格納されている


def main(process):  # メイン関数
    folder = ["R24B0", "R12B12", "R0B24", "RB24", "RB12"]  # 実験条件
    labelcolor = ["red", "purple", "blue", "black", "gray"]

    if __name__ != 'efd':
        process = int(input(
            "Select process\n 3: All process\n 2: Only before PCA\n 1: Only after Fourier\n Others: None\n"))

    if (1 < process < 4):  # 実験条件毎にdirectory移動
        print("=====start_Fourier=====")
        for now_folder in folder:
            path = "./{0}".format(now_folder)
            os.chdir(path)
            main_procedure(now_folder)
            os.chdir("../")
        print("=====*fin*_Fourier=====")

    if (0 < process < 4 and process % 2 == 1):
        print("=====start_PCA=====")
        N = 10  # 10級数まで計算
        M = N * 4 - 3  # d1とEの有無で調節
        np_pca_dataset = np.empty((0, M + 2), float)  # Eとnameのデータも追加
        len_list = []
        for now_folder in folder:
            path = "./{0}".format(now_folder)
            os.chdir(path)
            np_dataset = matrix_procedure(now_folder, N, M)
            np_pca_dataset = np.append(np_pca_dataset, np_dataset, axis=0)
            len_list.append(len(np_dataset))
            os.chdir("../")

        print(len_list)
        print(np_pca_dataset.shape)
        pca_procedure(np_pca_dataset, len_list, N, "all")  # 全ての条件でPCA
        pca_all_procedure(np_pca_dataset, len_list, N,
                          folder, labelcolor)  # 一斉に描画用

        num_start = 0
        num_fin = 0
        for i in range(len(folder)):
            num_fin += len_list[i]
            print(len_list[i])
            print(np_pca_dataset[num_start:num_fin].shape)
            pca_procedure(
                np_pca_dataset[num_start:num_fin], [len_list[i]], N, folder[i])  # 全ての条件で
            num_start = num_fin

        coefficient_procedure(np_pca_dataset, len_list,
                              folder, labelcolor)  # 通常コメントアウト
        print("=====*fin*_PCA=====")


def main_procedure(now_folder):  # 工程1
    my_jpg_list = glob.glob("*.JPG")  # JPGの探索とループ
    for my_file in my_jpg_list:  # 画像からフーリエ係数の導出と正規化まで
        file_name = "{0}_{1}".format(now_folder, my_file[:-4])
        img = cv2.imread(my_file)

        print("-----start_" + file_name + "-----")
        path = "./{0}".format(file_name)
        os.makedirs(path, exist_ok=True)
        os.chdir(path)

        print("|--- jpg > csv & graph")  # jpgから輪郭抽出
        obj = draw_contours(file_name, img)

        print("|--- list_diff > csv & graph")  # 葉毎に分割
        obj_div = division(file_name, obj.x_list, obj.y_list)

        for i in obj_div.leaf_number:
            path = "./{0}_{1}".format(file_name, i)
            os.makedirs(path, exist_ok=True)
            os.chdir(path)

            # フーリエ級数展開
            print("|--- fourier > graph @{0}_{1}".format(file_name, i))
            obj_frr = fourier(
                file_name, i, obj_div.np_time_cal[i], obj_div.np_x_cal[i], obj_div.np_y_cal[i])

            # 標準化
            print("|--- standardization > graph @{0}_{1}".format(file_name, i))
            obj_std = standardization(
                file_name, i, obj_div.np_time_cal[i], obj_frr.np_AB_set, obj_frr.N_set)
            print(
                " |-- E:{0:.1f} theta:{1:.3f} phi:{2:.3f}".format(obj_std.E, obj_std.theta, obj_std.phi))

            os.chdir("../")

        print("-----*fin*_" + file_name + "-----")
        os.chdir("../")


def matrix_procedure(now_folder, N, M):  # 必要なデータをcsvから持ってきて行列にする
    files1 = os.listdir("./")
    files_dir1 = [f for f in files1 if os.path.isdir(os.path.join("./", f))]

    np_dataset = np.empty((0, M + 2), float)  # Eとnameのデータも追加
    for dirfile1 in files_dir1:
        print("-----start_" + dirfile1 + "-----")
        os.chdir("./{0}".format(dirfile1))
        files2 = os.listdir("./")
        files_dir2 = [f for f in files2 if os.path.isdir(
            os.path.join("./", f))]

        np_buffer = []
        np_pca_row = []
        E = 0.0
        for dirfile2 in files_dir2:
            print("|--- csv > np_array @{0}".format(dirfile2))
            os.chdir("./{0}".format(dirfile2))

            np_buffer = np.loadtxt(
                '{0}_std_frequency.csv'.format(dirfile2), delimiter=',')
            E = np.loadtxt('{0}_std_E.csv'.format(dirfile2), delimiter=',')[0]
            np_pca_row = np.hstack(
                [np.ravel(np_buffer[1:, 1:N]), np_buffer[4, 0], E, '{0}'.format(dirfile2)])  # d1 E name
            np_dataset = np.append(np_dataset, np.array([np_pca_row]), axis=0)

            os.chdir("../")

        print("-----*fin*_" + dirfile1 + "-----")
        os.chdir("../")

    return np_dataset


def pca_procedure(np_pca_dataset, len_list, N, conditions):  # 工程2 標準化>PCA>フーリエ逆変換>可視化
    np_pca_dataset_coef = np_pca_dataset[:,
                                         :-2].astype(np.float32)  # Eとnameは除く
    # 標準化
    np_pca_mean = np.mean(np_pca_dataset_coef, axis=0)
    np_pca_SD = np.std(np_pca_dataset_coef, axis=0)
    np_pca_std = (np_pca_dataset_coef - np_pca_mean) / np_pca_SD  # ここで標準化

    # PCA
    sum_len = sum(len_list)
    mat_pca_std = np.matrix(np_pca_std)
    sig = mat_pca_std.T * mat_pca_std / (sum_len - 1)  # 共分散行列
    np_eigenvalue, np_eigenvector = LA.eigh(sig)  # エルミート実対称行列 固有値 固有ベクトル
    np_eigenvalue_rank = np.sort(np_eigenvalue)[::-1]
    np_eigenvalue_rank = np.where(
        np_eigenvalue_rank < 0, 0, np_eigenvalue_rank)
    # データ数が変数より少ない場合、固有値が負になることがあるから次元を下げる
    np_eigenvector_rank = np_eigenvector[:, np.argsort(np_eigenvalue)[::-1]]
    # np_CR = (np_eigenvalue_rank / np.sum(np_eigenvalue_rank)) * 100  # 寄与率
    # np_CCR = (np.cumsum(np_eigenvalue_rank) / np.sum(np_eigenvalue_rank)) * 100  # 累積寄与率
    # print("寄与率: {0}".format(np_CR))
    # print("累積寄与率: {0}".format(np_CCR))

    np_score = np.matrix(np_eigenvector_rank).T * \
        np.array([np_pca_mean]).T  # 主成分分析のスコアを出す
    np_new_mean = np.hstack([np_score, np_score, np_score])  # pm2SD用に3行にする
    np_new_SD = np.vstack([2 * np.sqrt(np_eigenvalue_rank),
                           np.zeros(len(np_pca_mean)), -2 * np.sqrt(np_eigenvalue_rank)]).T
    # print(2 * np.sqrt(np_eigenvalue_rank))

    N_PC = 5  # 第 N_PC 主成分まで計算
    a_1 = np.array([np.ones(3)]).T
    b_1 = np.array([np.zeros(3)]).T
    c_1 = np.array([np.zeros(3)]).T
    for i in range(N_PC):  # フーリエ逆変換 & 可視化
        i_one = np.array([np.zeros(len(np_score))]).T
        i_one[i] = 0.02  # 標準偏差をちょうどいい大きさに調整
        np_new_score = np_new_mean + np_new_SD * i_one  # pm2SDを第i主成分のみ残して計算
        np_new_fourier = np.matrix(
            np_eigenvector_rank) * np_new_score  # フーリエ級数を計算

        np_set_a = np.hstack([a_1, np_new_fourier.T[:, 0:N - 1]])  # 再構築a
        np_set_b = np.hstack(
            [b_1, np_new_fourier.T[:, N - 1:2 * (N - 1)]])  # 再構築b
        np_set_c = np.hstack(
            [c_1, np_new_fourier.T[:, 2 * (N - 1):3 * (N - 1)]])  # 再構築c
        np_set_d = np.hstack(
            [np_new_fourier.T[:, -1], np_new_fourier.T[:, 3 * (N - 1):-1]])  # 再構築d

        T = 2 * np.pi  # 再構築の際の周期は2pi
        np_new_t = np.array([np.arange(0, T, 0.01)])  # 時間0-2pi 0.01刻み
        np_new_N = np.array([np.arange(1, N + 1)]).T  # 級数の数
        np_new_cos = np.cos(np_new_N * np_new_t)  # cos
        np_new_sin = np.sin(np_new_N * np_new_t)  # sin

        # フーリエ逆変換
        np_new_x = np.array(np_set_a * np_new_cos + np_set_b * np_new_sin)
        np_new_y = np.array(np_set_c * np_new_cos + np_set_d * np_new_sin)

        # 可視化 InverseFourierTransform
        ax = plt.figure(num=0, dpi=120).gca()
        ax.set_title("IFT_PC:{0} {1}".format(
            i + 1, conditions), fontsize=14)  # 誤差あり
        # ax.set_title("IFT {0}".format(conditions), fontsize=14)  # 誤差なし
        ax.plot(np_new_x[0], np_new_y[0], linewidth=0.5,
                color="black", label="2SD")  # 誤差可視化
        ax.plot(np_new_x[2], np_new_y[2], linewidth=0.5,
                color="black", label="-2SD")  # 誤差可視化
        ax.plot(np_new_x[1], np_new_y[1], linewidth=2.5,
                color="green", label="mean")
        # plt.grid(which='major')
        # plt.legend()
        ax.set_xlabel('X axis', fontsize=14)
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylabel('Y axis', fontsize=14)
        ax.set_ylim([-2.5, 1.7])
        ax.set_aspect('equal', adjustable='box')
        plt.savefig("IFT_{0}_{1}.png".format(i + 1, conditions),
                    dpi=240, bbox_inches='tight', pad_inches=0.1)  # 誤差あり
        # plt.savefig("IFT_{0}.png".format(conditions), dpi=240, bbox_inches='tight', pad_inches=0.1)  # 誤差なし
        plt.pause(0.5)  # 計算速度を上げる場合はコメントアウト
        plt.clf()

    return


def pca_all_procedure(np_pca_dataset, len_list, N, folder, labelcolor):
    np_pca_dataset_coef = np_pca_dataset[:,
                                         :-1].astype(np.float32)  # nameは除く
    num_start = 0
    num_fin = 0
    ax = plt.figure(num=0, dpi=120).gca()
    ax.set_title("IFT mix", fontsize=14)
    for i in range(len(folder)):
        num_fin += len_list[i]

        # 標準化 保存も行う
        np_pca_mean = np.mean(np_pca_dataset_coef[num_start:num_fin], axis=0)
        np_pca_SD = np.std(np_pca_dataset_coef[num_start:num_fin], axis=0)
        print("{0}".format(folder[i]), np_pca_dataset[num_start:num_fin])
        print("{0}_mean".format(folder[i]), np_pca_mean)
        print("{0}_SD".format(folder[i]), np_pca_SD)
        obj_pca_raw = save_action("np_pca", "raw_{0}".format(folder[i]))
        obj_pca_raw.save_csv_1lists(np_pca_dataset[num_start:num_fin])
        obj_pca_mean = save_action("np_pca", "mean_{0}".format(folder[i]))
        obj_pca_mean.save_csv_1list(np_pca_mean)
        obj_pca_SD = save_action("np_pca", "SD_{0}".format(folder[i]))
        obj_pca_SD.save_csv_1list(np_pca_SD)
        np_pca_std = (
            np_pca_dataset_coef[num_start:num_fin, :-1] - np_pca_mean[:-1]) / np_pca_SD[:-1]  # ここで標準化 E除く

        # PCA
        sum_len = sum(len_list)
        mat_pca_std = np.matrix(np_pca_std)
        sig = mat_pca_std.T * mat_pca_std / (sum_len - 1)  # 共分散行列
        np_eigenvalue, np_eigenvector = LA.eigh(sig)  # エルミート実対称行列 固有値 固有ベクトル
        np_eigenvalue_rank = np.sort(np_eigenvalue)[::-1]
        np_eigenvalue_rank = np.where(
            np_eigenvalue_rank < 0, 0, np_eigenvalue_rank)
        # データ数が変数より少ない場合、固有値が負になることがあるから次元を下げる
        np_eigenvector_rank = np_eigenvector[:, np.argsort(np_eigenvalue)[
            ::-1]]

        np_score = np.matrix(np_eigenvector_rank).T * \
            np.array([np_pca_mean[:-1]]).T  # 主成分分析のスコアを出す
        np_new_mean = np.hstack([np_score, np_score, np_score])  # pm2SD用に3行にする
        np_new_SD = np.vstack([2 * np.sqrt(np_eigenvalue_rank),
                               np.zeros(len(np_pca_mean[:-1])), -2 * np.sqrt(np_eigenvalue_rank)]).T

        N_PC = 1  # 第1主成分だけ借りて計算
        a_1 = np.array([np.ones(3)]).T
        b_1 = np.array([np.zeros(3)]).T
        c_1 = np.array([np.zeros(3)]).T
        for j in range(N_PC):  # フーリエ逆変換 & 可視化
            i_one = np.array([np.zeros(len(np_score))]).T
            i_one[j] = 0.02  # 標準偏差をちょうどいい大きさに調整
            np_new_score = np_new_mean + np_new_SD * i_one  # pm2SDを第i主成分のみ残して計算
            np_new_fourier = np.matrix(
                np_eigenvector_rank) * np_new_score  # フーリエ級数を計算

            np_set_a = np.hstack([a_1, np_new_fourier.T[:, 0:N - 1]])  # 再構築a
            np_set_b = np.hstack(
                [b_1, np_new_fourier.T[:, N - 1:2 * (N - 1)]])  # 再構築b
            np_set_c = np.hstack(
                [c_1, np_new_fourier.T[:, 2 * (N - 1):3 * (N - 1)]])  # 再構築c
            np_set_d = np.hstack(
                [np_new_fourier.T[:, -1], np_new_fourier.T[:, 3 * (N - 1):-1]])  # 再構築d

            T = 2 * np.pi  # 再構築の際の周期は2pi
            np_new_t = np.array([np.arange(0, T, 0.01)])  # 時間0-2pi 0.01刻み
            np_new_N = np.array([np.arange(1, N + 1)]).T  # 級数の数
            np_new_cos = np.cos(np_new_N * np_new_t)  # cos
            np_new_sin = np.sin(np_new_N * np_new_t)  # sin

            # フーリエ逆変換
            np_new_x = np.array(np_set_a * np_new_cos + np_set_b * np_new_sin)
            np_new_y = np.array(np_set_c * np_new_cos + np_set_d * np_new_sin)

            # 可視化 InverseFourierTransform
            ax.plot(np_new_x[1], np_new_y[1], linewidth=1.5,
                    color=labelcolor[i], label="{0}".format(folder[i]))

        num_start = num_fin

    # plt.grid(which='major')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax.set_xlabel('X axis', fontsize=14)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylabel('Y axis', fontsize=14)
    ax.set_ylim([-2.5, 1.7])
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("IFT_mix.png", dpi=240,
                bbox_inches='tight', pad_inches=0.1)  # 誤差なし
    plt.pause(0.5)  # 計算速度を上げる場合はコメントアウト
    plt.clf()

    return


def coefficient_procedure(np_pca_dataset, len_list, folder, labelcolor):  # フーリエ係数推移可視化
    np_pca_dataset_coef = np_pca_dataset[:, :-1].astype(np.float32)  # nameは除く
    coefficient_list = []
    num_start = 0
    num_fin = 0
    # print(folder)
    # print(len_list)
    for i in range(len(folder)):
        print(i)
        num_fin += len_list[i]
        print(num_fin)
        coefficient_buff = np.mean(
            np_pca_dataset_coef[num_start:num_fin], axis=0)
        coefficient_list.append(coefficient_buff)
        num_start = num_fin

    xlabels_a = ["a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"]
    xlabels_b = ["b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"]
    xlabels_c = ["c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
    xlabels_d = ["d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10"]
    xlabels = []
    xlabels.extend(xlabels_a)
    xlabels.extend(xlabels_b)
    xlabels.extend(xlabels_c)
    xlabels.extend(xlabels_d)
    xlabels.append("d1/10")
    xlabels.append("E/1000")

    x_axis = np.arange(1, 39)

    # 可視化
    ax = plt.figure(num=0, dpi=120).gca()
    ax.set_title("Transition of Fourier Coefficients & E", fontsize=14)
    for i in range(len(folder)):
        y_axis = coefficient_list[i]
        y_axis[-2] /= 10
        y_axis[-1] /= 1000
        ax.plot(x_axis, y_axis, linewidth=1.5, color="{0}".format(
            labelcolor[i]), label="{0}".format(folder[i]))
        ax.scatter(x_axis[0], y_axis[0], s=2.5, color="black")
        ax.scatter(x_axis[-1], y_axis[-1], s=2.5, color="black")
    plt.plot([0.5, 0.5], [-0.20, 0.35], linewidth=0.7, color="black")
    plt.plot([9.5, 9.5], [-0.20, 0.35], linewidth=0.7, color="black")
    plt.plot([18.5, 18.5], [-0.20, 0.35], linewidth=0.7, color="black")
    plt.plot([27.5, 27.5], [-0.20, 0.35], linewidth=0.7, color="black")
    plt.plot([37.5, 37.5], [-0.20, 0.35], linewidth=0.7, color="black")
    plt.grid(True)
    plt.legend()
    ax.set_xlabel('Fourier Coefficients & E', fontsize=14)
    plt.xticks(x_axis, rotation=90)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel('Intensity', fontsize=14)
    # ax.set_aspect('equal', adjustable='box')
    plt.ylim([-0.20, 0.35])
    plt.savefig("TFC.png", dpi=360, bbox_inches='tight', pad_inches=0.1)
    plt.pause(0.5)  # 計算速度を上げる場合はコメントアウト
    plt.clf()

    return


class division:  # リスト内を微分して分割
    def __init__(self, file_name, x_list, y_list):
        self.file_name = file_name
        self.np_x = np.array(x_list)  # numpy化
        self.np_y = np.array(y_list)  # numpy化
        self.leaf_number = []
        self.np_time_cal = []  # 後にnumpy化
        self.np_x_cal = []  # 後にnumpy化
        self.np_y_cal = []  # 後にnumpy化
        self.list_diff()
        obj_div_csv = save_action(self.file_name, "split")  # csvに保存
        obj_div_csv.save_csv_3lists(
            self.np_time_cal, self.np_x_cal, self.np_y_cal)

    def list_diff(self):  # 分割
        np_x_diff = np.diff(self.np_x, n=1)  # 差分計算
        np_y_diff = np.diff(self.np_y, n=1)  # 差分計算

        time_base = np.sqrt(np.power(np_x_diff, 2) + np.power(np_y_diff, 2))
        time_0_base = np.concatenate(
            [np.zeros(1), time_base], axis=0)  # 移動距離の計算
        np_time = np.cumsum(time_0_base)

        np_x_diff_degi = np.where(np.abs(np_x_diff) > 50, 1, 0)  # 葉毎での分割用
        index = np.where(np_x_diff_degi > 0)

        index_leaf_number = index[0] + 1
        self.np_time_cal = np.split(np_time, index_leaf_number)
        self.np_x_cal = np.split(self.np_x, index_leaf_number)
        self.np_y_cal = np.split(self.np_y, index_leaf_number)

        for i in range(len(index_leaf_number) + 1):
            self.np_time_cal[i] = self.np_time_cal[i] - \
                self.np_time_cal[i][0]  # 初期値を0に調整
            self.np_x_cal[i] = self.np_x_cal[i] - \
                self.np_x_cal[i][0]  # 初期値を0に調整
            self.np_y_cal[i] = self.np_y_cal[i] - \
                self.np_y_cal[i][0]  # 初期値を0に調整

            if self.np_time_cal[i][-1] < 500:  # 細かい輪郭は飛ばす # 20201105 1500>>500
                continue

            self.leaf_number.append(i)
            path = "./{0}_{1}".format(self.file_name, i)
            os.makedirs(path, exist_ok=True)
            os.chdir(path)

            obj_div_graph = draw_graph_XY_tX_tY(
                self.file_name, i, "nom")  # グラフ化
            obj_div_graph.single_graph(
                self.np_x_cal[i], self.np_y_cal[i], "X Axis", "Y Axis", "XY")
            obj_div_graph.single_graph(
                self.np_time_cal[i], self.np_x_cal[i], "time", "X Axis", "TX")
            obj_div_graph.single_graph(
                self.np_time_cal[i], self.np_y_cal[i], "time", "Y Axis", "TY")

            os.chdir("../")


class draw_contours:  # 色調に差があり、輪郭になる場合HSVに変換>>>2値化して判別
    def __init__(self, file_name, img):
        self.file_name = file_name
        self.img = img
        self.x_list = []
        self.y_list = []
        self.trimming()
        self.back_revision()  # 20201105追加
        self.hsv_transration()
        self.gauss_transration()
        self.hsv_binary()
        self.contour_extraction()
        self.graph_and_csv()

    def trimming(self):  # 周囲をトリミング
        height, width = self.img.shape[:2]
        self.dst = self.img[200:height - 100, 50:width - 50]  # トリミングの大きさ
        dst = save_action(self.file_name, "dst")
        dst.save_img(self.dst)

    def back_revision(self):  # 背景のグラデーションを直す 20201105
        lab = cv2.cvtColor(self.dst, cv2.COLOR_BGR2LAB)  # G BRからLABに変換
        lab_planes = cv2.split(lab)  # LABに分離
        # L(明度)に対してGray画像と同様な抽出・処理を実施
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        lab_planes[0] = clahe.apply(lab_planes[0])  # L(明度)に対して明るくする
        lab = cv2.merge(lab_planes)  # LABをマージ
        self.bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # LABからBGRに変換
        bgr = save_action(self.file_name, "bgr")
        bgr.save_img(self.bgr)

    def hsv_transration(self):  # 色調変換
        self.hsv = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)
        hsv = save_action(self.file_name, "hsv")
        hsv.save_img(self.hsv)

    def gauss_transration(self):  # ガウス変換
        self.gauss = cv2.GaussianBlur(self.hsv, (15, 15), 3)  # フィルタの大きさ
        gauss = save_action(self.file_name, "gauss")
        gauss.save_img(self.gauss)

    def hsv_binary(self):  # HSV制限2値化
        lower = np.array([22, 100, 90])  # 下限 32 32 90
        upper = np.array([76, 255, 255])  # 上限 108 255 240
        self.img_HSV = cv2.inRange(self.gauss, lower, upper)
        img_HSV = save_action(self.file_name, "img_HSV")
        img_HSV.save_img(self.img_HSV)

    def contour_extraction(self):  # 輪郭抽出
        contours, hierarchy = cv2.findContours(
            self.img_HSV, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.boundingbox = self.dst
        for i in range(0, len(contours)):
            if len(contours[i]) > 0:
                if cv2.contourArea(contours[i]) < 1000:  # 20201105 500>>1000
                    continue
                cv2.polylines(self.boundingbox,
                              contours[i], True, (255, 255, 255), 5)
                buf_np = contours[i].flatten()  # numpyの多重配列>>>展開する
                for i, elem in enumerate(buf_np):
                    if i % 2 == 0:
                        self.x_list.append(elem)
                    else:
                        self.y_list.append(elem * (-1))
        boundingbox = save_action(self.file_name, "boundingbox")
        boundingbox.save_img(self.boundingbox)

    def graph_and_csv(self):  # graph & csv
        obj_graph = draw_graph(
            self.file_name, self.x_list, self.y_list)  # グラフ化
        obj_graph.contour_graph()

        obj_csv = save_action(self.file_name, "contours")  # csvに保存
        obj_csv.save_csv_1lists([self.x_list, self.y_list])


class draw_graph:  # グラフの描画
    def __init__(self, file_name, x_list, y_list):
        self.file_name = file_name
        self.x_list = x_list
        self.y_list = y_list
        self.ax = plt.figure(num=0, dpi=120).gca()

    def contour_graph(self):  # 輪郭座標のグラフ化
        self.graph()
        self.ax.set_xlabel('X Axis', fontsize=14)
        self.ax.set_ylabel('Y Axis', fontsize=14)
        self.ax.set_aspect('equal', adjustable='box')
        graph = save_action(self.file_name + "_contours", None)
        graph.save_graph()

    def graph(self):  # グラフ描画の基本を集約
        self.ax.set_title(self.file_name, fontsize=14)
        self.ax.scatter(self.x_list, self.y_list, s=1,
                        color="red", label=self.file_name)
        self.ax.plot(self.x_list, self.y_list, linewidth=1)
        plt.grid(which='major')


class draw_graph_multi(draw_graph):  # グラフの描画を継承 複数のデータ>>>1枚
    def __init__(self, file_name, i, save_name):
        print(" |-- now_{0}_{1}".format(i, save_name))
        self.sub_file_name = file_name
        self.i = i
        self.save_name = save_name

    def multi_graph(self, np_n, np_set, name_set, labelx, labely):  # 複数のデータを1枚のグラフに
        super().__init__(self.sub_file_name +
                         "_{0}_{1}".format(self.i, self.save_name), np_n, np_set)

        self.ax.set_title(self.file_name, fontsize=14)  # super().graph()が厳しい
        for j in range(len(name_set)):
            self.ax.plot(
                self.x_list, self.y_list[j], linewidth=1, label="{0}".format(name_set[j]))
        plt.grid(which='major')

        self.ax.set_xlabel(labelx, fontsize=14)
        self.ax.set_ylabel(labely, fontsize=14)
        if (self.save_name[-3:] == "Log"):
            plt.xscale("log")
            plt.grid(which='minor')
        if (self.save_name[:3] == "std"):
            plt.ylim([-1.0, 2.0])
            plt.legend()
        graph = save_action(self.file_name, None)
        graph.save_graph()


class draw_graph_XY_tX_tY(draw_graph):  # グラフの描画を継承 3軸を2軸で表現
    def __init__(self, file_name, i, save_name):
        print(" |-- now_{0}_{1}".format(i, save_name))
        self.sub_file_name = file_name
        self.i = i
        self.save_name = save_name

    def single_graph(self, x_list, y_list, labelx, labely, labelaxis):
        super().__init__(self.sub_file_name +
                         "_{0}_{1}_{2}".format(labelaxis, self.i, self.save_name), x_list, y_list)
        super().graph()
        self.ax.set_xlabel(labelx, fontsize=14)
        self.ax.set_ylabel(labely, fontsize=14)
        if (labelaxis == "XY"):
            self.ax.set_aspect('equal', adjustable='box')
        if (self.save_name[:3] == "std"):
            plt.ylim([-3.0, 3.0])
            if (labelaxis == "XY"):
                plt.xlim([-2.0, 2.0])
            # else:
            # plt.xlim([-3.5, 3.5])
        graph = save_action(self.file_name, None)
        graph.save_graph()


class fourier:  # フーリエ級数展開
    def __init__(self, file_name, i, np_time, np_x, np_y):
        self.pi = np.pi
        self.file_name = file_name
        self.i = i
        self.np_t = np_time
        self.T = self.np_t[-1]  # 0>>>Tの積分
        # フーリエ級数展開 N = 1, 3, 5, 10, 50, 100
        self.N_set = [1, 3, 5, 10, 50, 100]
        self.N = self.N_set[-1]  # 周波数100番目まで計算
        self.np_n_ver = np.arange(1, self.N + 1)  # 横ベクトルで作成
        self.np_n = np.array([self.np_n_ver]).T  # 縦ベクトルに変更
        self.np_Ax, self.np_Bx = self.fourier_cal(np_x)  # フーリエ級数展開 tX
        self.np_Ay, self.np_By = self.fourier_cal(np_y)  # フーリエ級数展開 tY
        self.np_AB_set = np.array(
            [self.np_Ax, self.np_Bx, self.np_Ay, self.np_By])
        # self.coefficient_graph()  # standardizationにも使用
        self.np_fx_set, self.np_fy_set = self.before_sigma(
            self.np_Ax, self.np_Bx, self.np_Ay, self.np_By)
        # self.fourier_graph()  # standardizationにも使用
        obj_frr_F_csv = save_action(
            self.file_name, "{0}_fourier_frequency".format(self.i))  # csvに保存_周波数領域
        obj_frr_F_csv.save_csv_1list_1lists(self.np_n_ver, self.np_AB_set)
        obj_frr_csv = save_action(
            self.file_name, "{0}_fourier_set".format(self.i))  # csvに保存_時間領域
        obj_frr_csv.save_csv_1list_2lists(
            self.np_t, self.np_fx_set, self.np_fy_set)

    def before_sigma(self, np_Ax_origin, np_Bx_origin, np_Ay_origin, np_By_origin):  # 係数>>>和の前まで
        Axcos = np.array([np_Ax_origin]).T * np.cos(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Bxsin = np.array([np_Bx_origin]).T * np.sin(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Aycos = np.array([np_Ay_origin]).T * np.cos(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Bysin = np.array([np_By_origin]).T * np.sin(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        np_fx_set_origin = Axcos + Bxsin
        np_fy_set_origin = Aycos + Bysin
        return np_fx_set_origin, np_fy_set_origin

    def coefficient_graph(self):  # フーリエ係数確認のために描画
        name_AB_set = ["x_cos", "x_sin", "y_cos", "y_sin"]
        obj_frr_graph = draw_graph_multi(
            self.file_name, self.i, "fourier_{0}_frequencyLog".format(self.N))  # グラフ化
        obj_frr_graph.multi_graph(
            self.np_n_ver, self.np_AB_set, name_AB_set, "n Frequency", "Intensity")

    def fourier_cal(self, np_f):  # cosの係数Aとsinの係数B
        ftcos = np_f * np.cos(2 * self.pi * self.np_n * self.np_t / self.T)
        ftsin = np_f * np.sin(2 * self.pi * self.np_n * self.np_t / self.T)
        # A0 = 2 * self.integral(np_f) / self.T  # A0の項の計算
        np_A = 2 * self.integral(ftcos) / self.T  # フーリエ級数展開cos係数(積分)
        np_B = 2 * self.integral(ftsin) / self.T  # フーリエ級数展開sin係数(積分)
        return np_A, np_B

    def fourier_graph(self):  # フーリエ級数展開確認のために描画
        for j in self.N_set:
            np_fx = np.sum(self.np_fx_set[:j], axis=0)
            np_fy = np.sum(self.np_fy_set[:j], axis=0)
            obj_fourier_graph = draw_graph_XY_tX_tY(
                self.file_name, self.i, "fourier_{0}".format(j))
            obj_fourier_graph.single_graph(
                np_fx, np_fy, "X Axis", "Y Axis", "XY")
            obj_fourier_graph.single_graph(
                self.np_t, np_fx, "time", "X Axis", "TX")
            obj_fourier_graph.single_graph(
                self.np_t, np_fy, "time", "Y Axis", "TY")

    def integral(self, f):  # 台形則で積分 sigma(fdt)
        del_f = f[0:, 1:] + f[0:, :-1]
        del_np_t = self.np_t[1:] - self.np_t[:-1]
        return np.sum((del_f * del_np_t) / 2.0, axis=1)


class save_action:  # 保存動作を集約
    def __init__(self, file_name, save_name):
        self.file_name = file_name
        self.save_name = save_name

    def save_csv_1list(self, list):  # list1つのみ csv
        with open("{0}_{1}.csv".format(self.file_name, self.save_name), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(list)

    def save_csv_1list_1lists(self, list, lists):  # list1つ lists1つ csv
        with open("{0}_{1}.csv".format(self.file_name, self.save_name), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(list)
            writer.writerows(lists)

    def save_csv_1list_2lists(self, list, fx_lists, fy_lists):  # list1つ lists2つ csv
        with open("{0}_{1}.csv".format(self.file_name, self.save_name), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(list)
            writer.writerow("\n")
            writer.writerows(fx_lists)
            writer.writerow("\n")
            writer.writerows(fy_lists)

    def save_csv_1lists(self, lists):  # lists1つのみ csv
        with open("{0}_{1}.csv".format(self.file_name, self.save_name), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(lists)

    # lists3つ t_list x_list y_list csv
    def save_csv_3lists(self, t_lists, x_lists, y_lists):
        with open("{0}_{1}.csv".format(self.file_name, self.save_name), "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(t_lists)
            writer.writerow("\n")
            writer.writerows(x_lists)
            writer.writerow("\n")
            writer.writerows(y_lists)

    def save_graph(self):  # グラフの保存
        plt.savefig("{0}.png".format(self.file_name), dpi=240,
                    bbox_inches='tight', pad_inches=0.1)
        # plt.pause(0.3)  # 計算速度を上げる場合はコメントアウト
        plt.clf()

    def save_img(self, image):  # 画像の保存
        cv2.imwrite("{0}_{1}.jpg".format(
            self.file_name, self.save_name), image)


class standardization:  # fourierの後の標準化
    def __init__(self, file_name, i, np_time, np_AB_set_origin, N_set_origin):
        self.pi = np.pi
        self.file_name = file_name
        self.i = i
        self.np_t = np_time
        self.T = self.np_t[-1]  # 0>>>Tの積分
        self.np_AB_set_origin = np_AB_set_origin
        self.N_set = N_set_origin
        self.N = self.N_set[-1]
        self.np_n_ver = np.arange(1, self.N + 1)  # 横ベクトルで作成
        self.np_n = np.array([self.np_n_ver]).T  # 縦ベクトルに変更
        self.np_AB1 = np.array([np_AB_set_origin[:, 0]]).T  # n_1の縦ベクトル
        self.theta = 0.0  # 始点 0>>>pi
        self.corrected_start()
        self.E = 0.0  # 大きさ
        self.a1_1 = 0.0
        self.c1_1 = 0.0
        self.corrected_size()
        self.phi = 0.0  # 回転 0>>>pi
        self.rotation()
        obj_std_size_csv = save_action(
            self.file_name, "{0}_std_E".format(self.i))  # csvに大きさ保存
        obj_std_size_csv.save_csv_1list([self.E, self.theta, self.phi])
        self.np_AB_std = []  # 係数の標準化
        self.calculation_n()
        self.coefficient_graph()  # fourier係数の可視化
        self.np_fx_std, self.np_fy_std = self.before_sigma(
            self.np_AB_std[0], self.np_AB_std[1], self.np_AB_std[2], self.np_AB_std[3])
        self.fourier_graph()  # 結果の可視化
        obj_std_F_csv = save_action(
            self.file_name, "{0}_std_frequency".format(self.i))  # csvに保存_周波数領域
        obj_std_F_csv.save_csv_1list_1lists(self.np_n_ver, self.np_AB_std)
        obj_std_csv = save_action(
            self.file_name, "{0}_std_set".format(self.i))  # csvに保存_時間領域
        obj_std_csv.save_csv_1list_2lists(
            self.np_t, self.np_fx_std, self.np_fy_std)

    def before_sigma(self, np_Ax_origin, np_Bx_origin, np_Ay_origin, np_By_origin):  # 係数>>>和の前まで
        Axcos = np.array([np_Ax_origin]).T * np.cos(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Bxsin = np.array([np_Bx_origin]).T * np.sin(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Aycos = np.array([np_Ay_origin]).T * np.cos(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        Bysin = np.array([np_By_origin]).T * np.sin(2 *
                                                    self.pi * self.np_n * self.np_t / self.T)
        np_fx_set_origin = Axcos + Bxsin
        np_fy_set_origin = Aycos + Bysin
        return np_fx_set_origin, np_fy_set_origin

    def calculation_n(self):
        cos_n_theta = np.cos(self.np_n_ver * self.theta)
        sin_n_theta = np.sin(self.np_n_ver * self.theta)
        np_n_theta = np.array(
            [cos_n_theta, (-1) * sin_n_theta, sin_n_theta, cos_n_theta])
        buffer_cal = self.matrix_multiplication(
            self.np_AB_set_origin, np_n_theta)
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        np_phi = np.array([[cos_phi, sin_phi, (-1) * sin_phi, cos_phi]]).T
        self.np_AB_std = self.matrix_multiplication(
            np_phi, buffer_cal) / self.E
        if (self.np_AB_std[0, 0] < -0.5):  # a1で逆位相の修正
            self.np_AB_std *= -1

    def coefficient_graph(self):  # フーリエ係数確認のために描画
        name_AB_set = ["x_cos", "x_sin", "y_cos", "y_sin"]
        obj_frr_graph = draw_graph_multi(
            self.file_name, self.i, "std_{0}_frequencyLog".format(self.N))  # グラフ化
        obj_frr_graph.multi_graph(
            self.np_n_ver, self.np_AB_std, name_AB_set, "n Frequency", "Intensity")

    def corrected_size(self):
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        np_theta = np.array(
            [[cos_theta, (-1) * sin_theta, sin_theta, cos_theta]]).T
        np_size = self.matrix_multiplication(self.np_AB1, np_theta)
        self.a1_1 = np_size[0, 0]
        self.c1_1 = np_size[2, 0]
        self.E = np.sqrt(self.a1_1 ** 2 + self.c1_1 ** 2)

    def corrected_start(self):
        numer = 2 * (self.np_AB1[0, 0] * self.np_AB1[1,
                                                     0] + self.np_AB1[2, 0] * self.np_AB1[3, 0])
        denom = ((self.np_AB1[0, 0]) ** 2 + (self.np_AB1[2, 0]) **
                 2) - ((self.np_AB1[1, 0]) ** 2 + (self.np_AB1[3, 0]) ** 2)
        self.theta = np.arctan(numer / denom) / 2.0 - self.pi / 2.0
        if (self.theta > 0):
            self.theta -= self.pi / 2.0
        elif (self.theta < -1 * self.pi / 2.0):
            self.theta += self.pi / 2.0

    def fourier_graph(self):  # フーリエ級数展開確認のために描画
        for j in self.N_set:
            np_fx = np.sum(self.np_fx_std[:j], axis=0)
            np_fy = np.sum(self.np_fy_std[:j], axis=0)
            obj_fourier_graph = draw_graph_XY_tX_tY(
                self.file_name, self.i, "std_{0}".format(j))
            obj_fourier_graph.single_graph(
                np_fx, np_fy, "X Axis", "Y Axis", "XY")
            obj_fourier_graph.single_graph(
                self.np_t, np_fx, "time", "X Axis", "TX")
            obj_fourier_graph.single_graph(
                self.np_t, np_fy, "time", "Y Axis", "TY")

    def matrix_multiplication(self, np_A, np_B):  # 2*2の行列積を縦ベクトルや行列で返す 2次元
        return np_A[[0, 0, 2, 2]] * np_B[[0, 1, 0, 1]] + np_A[[1, 1, 3, 3]] * np_B[[2, 3, 2, 3]]

    def rotation(self):
        self.phi = np.arctan(self.c1_1 / self.a1_1)
        if (abs(self.phi) > self.pi / 4.0):  # phiがおかしい時修正
            self.theta -= self.pi / 2.0
            self.corrected_size()
            self.rotation()


if __name__ == '__main__':
    main(0)
