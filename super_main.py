import efd
import os


def main():  # ディレクトリの構造上必要なメイン関数
    super_folder = ["test1", "test2", "test3"]

    process = int(input(
        "Select process\n 3: All process\n 2: Only before PCA\n 1: Only after Fourier\n Others: None\n"))

    for now_super_folder in super_folder:
        print("=====start_{0}=====".format(now_super_folder))
        path = "./{0}".format(now_super_folder)
        os.chdir(path)
        efd.main(process)
        os.chdir("../")
        print("=====*fin*_{0}=====".format(now_super_folder))

    TimeShiftAnalysis()


def TimeShiftAnalysis():
    print("=====TimeShiftAnalysis=====")


if __name__ == '__main__':
    main()
