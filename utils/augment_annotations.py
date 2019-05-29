from pandas import read_csv


def get_least_most_rep_class(annotations):
	"""Gets the least and most represented class for all labels

	:param annotations: A pandas data frame of the annotations
	:return: A tuple of the most and least represented class (string)
	"""
	val_dict = {
		' V_Neg': annotations[' V_Neg'].sum(),
		' V_Neut': annotations[' V_Neut'].sum(),
		' V_Pos': annotations[' V_Pos'].sum()
	}

	aro_dict = {
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
	weighted_biggest_aro = float(annotations[biggest_aro].sum()/total)/(1./2.)
	weighted_biggest_gam = float(annotations[biggest_gam].sum()/total)/(1./8.)

	if(weighted_biggest_val > weighted_biggest_aro) and (weighted_biggest_val > weighted_biggest_gam):
		biggest = biggest_val
	elif weighted_biggest_aro > weighted_biggest_gam:
		biggest = biggest_aro
	else:
		biggest = biggest_gam

	return smallest, biggest


def main():
	"""Takes a csv of annotations and performs oversampling to help balence the data
	"""
	annotations = read_csv("../train.csv")
	limit = 5517

	for i in range(0, limit):
		
		least_class, most_class = get_least_most_rep_class(annotations)

		data = annotations.loc[(annotations[least_class] == 1) & (annotations[most_class] == 0)]
		if len(data.index) == 0:
			print("Exiting Early at ", i)
			break
		samp = data.sample(1)
		annotations = annotations.append(samp)

	annotations.to_csv("train_augmented.csv")

if __name__ == "__main__":
	main()
