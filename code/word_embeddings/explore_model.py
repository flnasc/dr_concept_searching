import gensim
pre = [('symbolisme', 0.741622805595398), ('mythe', 0.7170282602310181), ('schème', 0.5630180835723877), ('langage', 0.5450016260147095), ('péché', 0.5371381044387817), ('sacré', 0.5355600118637085), ('mal', 0.5329419374465942), ('sens', 0.5250318050384521), ('mouvement', 0.5240223407745361), ('serpent', 0.5239921808242798), ('rêve', 0.5237582921981812), ('mystère', 0.5165060758590698), ('signe', 0.5121667385101318), ('mot', 0.5081802010536194), ('banquet', 0.5023268461227417), ('rite', 0.49562567472457886), ('texte', 0.4927329421043396), ('chiffre', 0.4908056855201721), ('signifié', 0.4872182607650757), ('désir', 0.48637616634368896)]
prelim = [('mythe', 0.5155792236328125), ('symbolisme', 0.49310243129730225), ('sens', 0.40665197372436523), ('langage', 0.39813467860221863), ('serpent', 0.3636723756790161), ('mal', 0.34314244985580444), ('péché', 0.3401647210121155), ('schème', 0.3365691900253296), ('symbolique', 0.33058133721351624), ('mot', 0.329504132270813), ('thème', 0.3256272077560425), ('chiffre', 0.32093432545661926), ('symptôme', 0.3208768963813782), ('désir', 0.3152922987937927), ('fantasme', 0.31499671936035156), ('mouvement', 0.31441396474838257), ('signifié', 0.31006914377212524), ('christ', 0.3087116479873657), ('texte', 0.3033600151538849), ('sacré', 0.302484393119812)]
def to_dict(l):
    return {term[0]: term[1] for term in l}

pre_dict = to_dict(pre)
prelim_dict = to_dict(prelim)

d0 = {'mythe':-0.499, 'symbole':-0.339, 'homme':-0.382, 'morale':-0.363, 'justice':-0.434}
d1 = {'mythe':-0.507, 'symbole':-0.368, 'homme':-0.450, 'morale':-0.443, 'justice':-0.452}
d2 = {'mythe':-0.472, 'symbole':-0.334, 'homme':-0.374,  'justice':-0.426}
d3 = {'mythe':-0.270, 'symbole':-0.213, 'homme':-0.212, 'morale':-0.334, 'justice':-0.372}
def p_change(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    common = d1_keys.intersection(d2_keys)
    print(len(common))
    percent_changes = []
    for key in common:
        change = d2[key] - d1[key]
        p = change/d1[key]
        percent_changes.append((key, p))

    return percent_changes


def avg(l):
    p_changes = [term[1] for term in l]
    return sum(p_changes)/len(p_changes)


#print(avg(p_change(prelim_dict, pre_dict)))
print(avg(p_change(d0, d2)))

if __name__ == '__main__':
    print('loading pretrained model')
    model = gensim.models.Word2Vec.load('../models/word2vec_french_2.model')
    print(model.iter)
    while True:
        try:
            word = input('Word: ').strip()
            print(model.most_similar(word, topn=20))
        except KeyError:
            continue
        except KeyboardInterrupt:
            quit(0)
