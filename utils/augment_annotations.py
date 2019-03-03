from pandas import read_csv


def calculate_class_weights(annotations):
	n_v_neg = annotations[' V_Neg'].sum()
	n_v_neut = annotations[' V_Neut'].sum()
	n_v_pos = annotations[' V_Pos'].sum()
	n_a_neg = annotations[' A_Neg'].sum()
	n_a_neut = annotations[' A_Neut'].sum()
	n_a_pos = annotations[' A_Pos'].sum()

	tot = n_v_neg + n_v_neut + n_v_pos

	print(float(n_v_neg/tot), " " , float(n_v_neut/tot), " ", float(n_v_pos/tot), " ", float(n_a_neg/tot),
	      " ",  float(n_a_neut/tot), " ", float(n_a_pos/tot))

	n_laning = annotations[' Laning'].sum()
	n_shopping = annotations[' Shopping'].sum()
	n_returning = annotations[' Returning'].sum()
	n_roaming = annotations[' Roaming'].sum()
	n_fighting = annotations[' Fighting'].sum()
	n_pushing = annotations[' Pushing'].sum()
	n_defending = annotations[' Defending'].sum()
	n_dead = annotations[' Dead'].sum()

	tot = n_laning + n_shopping + n_returning + n_roaming + n_fighting + n_pushing + n_defending + n_dead

	print(float(n_laning/tot), " " , float(n_shopping/tot), " ", float(n_returning/tot), " ", float(n_roaming/tot),
	      " ",  float(n_fighting/tot), " ", float(n_pushing/tot), " ", float(n_defending/tot), " ", float(n_dead/tot))


def get_least_most_rep_class(annotations):
	val_dict = {
		' V_Neg': annotations[' V_Neg'].sum(),
		' V_Neut': annotations[' V_Neut'].sum(),
		' V_Pos': annotations[' V_Pos'].sum()
	}

	aro_dict = {
		' A_Neg': annotations[' A_Neg'].sum(),
		' A_Neut': annotations[' A_Neut'].sum(),
		' A_Pos': annotations[' A_Pos'].sum()
	}

	game_dict = {
		' Laning': annotations[' Laning'].sum(),
		' Shopping': annotations[' Shopping'].sum(),
		' Returning': annotations[' Returning'].sum(),
		' Roaming': annotations[' Roaming'].sum(),
		' Fighting': annotations[' Fighting'].sum(),
		' Pushing': annotations[' Pushing'].sum(),
		' Defending': annotations[' Defending'].sum(),
		' Dead': annotations[' Dead'].sum()
	}

	total = val_dict[' V_Neg']+val_dict[' V_Neut']+val_dict[' V_Pos']

	smallest_val = min(val_dict, key=val_dict.get)
	smallest_aro = min(aro_dict, key=aro_dict.get)
	smallest_gam = min(game_dict, key=game_dict.get)

	weighted_smallest_val = float(annotations[smallest_val].sum()/total)/(1./3.)
	weighted_smallest_aro = float(annotations[smallest_aro].sum()/total)/(1./3.)
	weighted_smallest_gam = float(annotations[smallest_gam].sum()/total)/(1./8.)

	if(weighted_smallest_val < weighted_smallest_aro) and (weighted_smallest_val < weighted_smallest_gam):
		smallest = smallest_val
	elif weighted_smallest_aro < weighted_smallest_gam:
		smallest = smallest_aro
	else:
		smallest = smallest_gam

	biggest_val = max(val_dict, key=val_dict.get)
	biggest_aro = max(aro_dict, key=aro_dict.get)
	biggest_gam = max(game_dict, key=game_dict.get)

	weighted_biggest_val = float(annotations[biggest_val].sum()/total)/(1./3.)
	weighted_biggest_aro = float(annotations[biggest_aro].sum()/total)/(1./3.)
	weighted_biggest_gam = float(annotations[biggest_gam].sum()/total)/(1./8.)

	if(weighted_biggest_val > weighted_biggest_aro) and (weighted_biggest_val > weighted_biggest_gam):
		biggest = biggest_val
	elif weighted_biggest_aro > weighted_biggest_gam:
		biggest = biggest_aro
	else:
		biggest = biggest_gam

	return smallest, biggest


def main():
	annotations = read_csv("../train.csv")
	annotations = annotations[annotations[' Misc'] != 1]
	calculate_class_weights(annotations)

	for i in range(0, 2758):
		
		least_class, most_class = get_least_most_rep_class(annotations)

		data = annotations.loc[(annotations[least_class] == 1) & (annotations[most_class] == 0)]
		if len(data.index) == 0:
			print("Exiting Early at ", i)
			break
		samp = data.sample(1)
		annotations = annotations.append(samp)

	calculate_class_weights(annotations)
	annotations.to_csv("train_augmented.csv")


main()
