from pandas import read_csv


def calculate_class_weights(labels):
    annotations = read_csv(labels)

    n_v_neg = annotations[' V_Neg'].sum()
    n_v_neut = annotations[' V_Neut'].sum()
    n_v_pos = annotations[' V_Pos'].sum()
    n_a_neg = annotations[' A_Neg'].sum()
    n_a_neut = annotations[' A_Neut'].sum()
    n_a_pos = annotations[' A_Pos'].sum()
    n_laning = annotations[' Laning'].sum()
    n_shopping = annotations[' Shopping'].sum()
    n_returning = annotations[' Returning'].sum()
    n_roaming = annotations[' Roaming'].sum()
    n_fighting = annotations[' Fighting'].sum()
    n_pushing = annotations[' Pushing'].sum()
    n_defending = annotations[' Defending'].sum()
    n_dead = annotations[' Dead'].sum()
    n_misc = annotations[' Misc'].sum()

    w_v_neg = float(3600)/(3 * n_v_neg)
    w_v_neut = float(3600)/(3 * n_v_neut)
    w_v_pos = float(3600)/(3 * n_v_pos)

    tot = w_v_neg + w_v_neut + w_v_pos

    w_v_neg = w_v_neg/tot
    w_v_neut = w_v_neut/tot
    w_v_pos = w_v_pos/tot

    w_a_neg = float(3600) / (3 * n_a_neg)
    w_a_neut = float(3600) / (3 * n_a_neut)
    w_a_pos = float(3600) / (3 * n_a_pos)

    tot = w_a_neg + w_a_neut + w_a_pos

    w_a_neg = w_a_neg / tot
    w_a_neut = w_a_neut / tot
    w_a_pos = w_a_pos / tot

    w_laning = float(3600) / (9 * n_laning)
    w_shopping = float(3600) / (9 * n_shopping)
    w_returning = float(3600) / (9 * n_returning)
    w_roaming = float(3600) / (9 * n_roaming)
    w_fighting = float(3600) / (9 * n_fighting)
    w_pushing = float(3600) / (9 * n_pushing)
    w_defending = float(3600) / (9 * n_defending)
    w_dead = float(3600) / (9 * n_dead)
    w_misc = float(3600) / (9 * n_misc)

    tot = w_laning + w_shopping + w_returning + w_roaming + w_fighting + w_pushing + w_defending + w_dead + w_misc

    w_laning = w_laning/tot
    w_shopping = w_shopping/tot
    w_returning = w_returning/tot
    w_roaming = w_roaming/tot
    w_fighting = w_fighting/tot
    w_pushing = w_pushing/tot
    w_defending = w_defending/tot
    w_dead = w_dead/tot
    w_misc = w_misc/tot

    # v_percentages = {0: float(n_v_neg)/float(3600), 1: float(n_v_neut)/float(3600), 2: float(n_v_pos)/float(3600)}
    # a_percentages = {0: float(n_a_neg)/float(3600), 1: float(n_a_neut)/float(3600), 2: float(n_a_pos)/float(3600)}
    # g_percentages = {0: float(n_laning)/float(3600), 1: float(n_shopping)/float(3600),
    # 				 2: float(n_returning)/float(3600), 3: float(n_roaming)/float(3600),
    # 				 4: float(n_fighting)/float(3600), 5: float(n_pushing)/float(3600),
    # 				 6: float(n_defending)/float(3600), 7: float(n_dead)/float(3600),
    # 				 8: float(n_misc)/float(3600) }

    # print(v_percentages)
    # print(a_percentages)
    # print(g_percentages)

    v_weights = {0: w_v_neg, 1: w_v_neut, 2: w_v_pos}
    a_weights = {0: w_a_neg, 1: w_a_neut, 2: w_a_pos}
    g_weights = {
        0: w_laning, 1: w_shopping, 2: w_returning, 3: w_roaming, 4: w_fighting,
        5: w_pushing, 6: w_defending, 7: w_dead, 8: w_misc
    }
    return v_weights, a_weights, g_weights