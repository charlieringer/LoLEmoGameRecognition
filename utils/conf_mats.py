def get_conf_matrx(model, data_gen, batch_size, n_videos, model_flag, outfile):
    if model_flag == "both":
        get_conf_matrx_both(model, data_gen, batch_size, n_videos, outfile)
    elif model_flag == "game":
        get_conf_matrx_game(model, data_gen, batch_size, n_videos, outfile)
    else:
        get_conf_matrx_face(model, data_gen, batch_size, n_videos, outfile)


def get_conf_matrx_both(model, data_gen, batch_size, n_videos, outfile):
    val_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]
    aro_cnf_mat = [[0 for _ in range(2)] for _ in range(2)]
    gam_cnf_mat = [[0 for _ in range(8)] for _ in range(8)]

    output_file = open(outfile, "w")

    for _ in range(int(n_videos/batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
                val_cnf_mat[res[0][i].argmax()][data[1][0][i].argmax()] += 1
                aro_cnf_mat[res[1][i].argmax()][data[1][1][i].argmax()] += 1
                gam_cnf_mat[res[2][i].argmax()][data[1][2][i].argmax()] += 1

    output_file.write("Val Conf Mat" + "\n")
    print("Val Conf Mat")
    for i in range(3):
        output_file.write(str(val_cnf_mat[i]) + "\n")
        print(val_cnf_mat[i])

    output_file.write("Aro Conf Mat" + "\n")
    print("Aro Conf Mat")
    for i in range(2):
        output_file.write(str(aro_cnf_mat[i]) + "\n")
        print(aro_cnf_mat[i])

    output_file.write("Game Conf Mat" + "\n")
    print("Game Conf Mat")
    for i in range(8):
        output_file.write(str(gam_cnf_mat[i]) + "\n")
        print(gam_cnf_mat[i])

    output_file.close()


def get_conf_matrx_game(model, data_gen, batch_size, n_videos, outfile):
    gam_cnf_mat = [[0 for _ in range(8)] for _ in range(8)]

    output_file = open(outfile, "w")

    for _ in range(int(n_videos / batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
            gam_cnf_mat[res[i].argmax()][data[1][0][i].argmax()] += 1

    output_file.write("Game Conf Mat" + "\n")
    print("Game Conf Mat")
    for i in range(8):
        output_file.write(str(gam_cnf_mat[i]) + "\n")
        print(gam_cnf_mat[i])

    output_file.close()


def get_conf_matrx_face(model, data_gen, batch_size, n_videos, outfile):
    val_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]
    aro_cnf_mat = [[0 for _ in range(2)] for _ in range(2)]

    output_file = open(outfile, "w")

    for _ in range(int(n_videos/batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
                val_cnf_mat[res[0][i].argmax()][data[1][0][i].argmax()] += 1
                aro_cnf_mat[res[1][i].argmax()][data[1][1][i].argmax()] += 1

    output_file.write("Val Conf Mat" + "\n")
    print("Val Conf Mat")
    for i in range(3):
        output_file.write(str(val_cnf_mat[i]) + "\n")
        print(val_cnf_mat[i])

    output_file.write("Aro Conf Mat" + "\n")
    print("Aro Conf Mat")
    for i in range(2):
        output_file.write(str(aro_cnf_mat[i]) + "\n")
        print(aro_cnf_mat[i])

    output_file.close()
