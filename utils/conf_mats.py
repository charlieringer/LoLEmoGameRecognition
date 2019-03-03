def get_conf_matrx(model, data_gen, batch_size, n_videos, model_flag):
    if model_flag == "both":
        get_conf_matrx_both(model, data_gen, batch_size, n_videos)
    elif model_flag == "game":
        get_conf_matrx_game(model, data_gen, batch_size, n_videos)
    else:
        get_conf_matrx_face(model, data_gen, batch_size, n_videos)


def get_conf_matrx_both(model, data_gen, batch_size, n_videos):
    val_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]
    aro_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]
    gam_cnf_mat = [[0 for _ in range(8)] for _ in range(8)]

    for _ in range(int(n_videos/batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
                val_cnf_mat[res[0][i].argmax()][data[1][0][i].argmax()] += 1
                aro_cnf_mat[res[1][i].argmax()][data[1][1][i].argmax()] += 1
                gam_cnf_mat[res[2][i].argmax()][data[1][2][i].argmax()] += 1

    print("Val Conf Mat")
    for i in range(3):
        print(val_cnf_mat[i])

    print("Aro Conf Mat")
    for i in range(3):
        print(aro_cnf_mat[i])

    print("Game Conf Mat")
    for i in range(8):
        print(gam_cnf_mat[i])


def get_conf_matrx_game(model, data_gen, batch_size, n_videos):
    gam_cnf_mat = [[0 for _ in range(8)] for _ in range(8)]

    for _ in range(int(n_videos / batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
            gam_cnf_mat[res[i].argmax()][data[1][0][i].argmax()] += 1

    print("Game Conf Mat")
    for i in range(8):
        print(gam_cnf_mat[i])


def get_conf_matrx_face(model, data_gen, batch_size, n_videos):
    val_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]
    aro_cnf_mat = [[0 for _ in range(3)] for _ in range(3)]

    for _ in range(int(n_videos/batch_size)):
        data = next(data_gen)
        res = model.predict(data[0], batch_size=batch_size)
        for i in range(batch_size):
                val_cnf_mat[res[0][i].argmax()][data[1][0][i].argmax()] += 1
                aro_cnf_mat[res[1][i].argmax()][data[1][1][i].argmax()] += 1

    print("Val Conf Mat")
    for i in range(3):
        print(val_cnf_mat[i])

    print("Aro Conf Mat")
    for i in range(3):
        print(aro_cnf_mat[i])
