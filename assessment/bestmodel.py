import json
import matplotlib.pyplot as plt

features = [
"cinsiyet",
 "yas",	
 "ilk-atak-yasi",	
 "Ates-ataklar-seklinde",	
 "Boyun-lenf-sislik",
 "aft-var",	
 "atese-tonsillofarenjit-eslik",	
 "atese-karin-agrisi",	
 "atese-eklem-sislik-kizariklik",	
 "atese-artralji",	
"atese-isal",	
"erizipel-kizariklik",	
"atesli-atak-kac-gün",	
"atak-yilda-kac-kez",	
"serum-amiloid-A",	
"CRP",	
"fibrinojen",	
"sedimantasyon",	
"lokosit",	
"ANS",	
"trombosit",	
"hemogram",	
"monosit-yuzde",	
"monosit-sayi",
"Tani"
]


f = open('/Users/nevil/projects/mlapi/data/project/1/best_model.json')
model = json.load(f)
f.close()
print("model=" +  str(len(model)))


# MAX
maxs = []
for i in model:
    maxs.append(i["max"])

max_of_model = max(maxs)

print("max=" + str(max_of_model))
print("max=" + str(maxs[-1]))



# PARAMETERS
parameters = []
for i in range(30):
    parameters.append(0)

for i in model:
    ps = i["parameter"]
    for p in ps:
        #print(str(p))
        #print(str(parameters[p]))
        parameters[p] = parameters[p] + 1

print("parameters=" + str(parameters))


# PARAMETER SET
paramset = []
for i in range(30):
    paramset.append([])

for i in model:
    ps = i["parameter"]
    paramset[len(ps)].append(ps)

print("paramset=" + str(paramset))


paramsetnames = []
for i in range(30):
    paramsetnames.append([])

for i in model:
    ps = i["parameter"]
    psnames = []
    for p in ps:
        psnames.append(features[p])
    paramsetnames[len(ps)].append(psnames)

print("paramsetnames=" + str(paramsetnames))


# ACCURACIES
accuracies = { "DecisionTreeClassifier":0, "GaussianNB":0,"LinearDiscriminantAnalysis":0,"KNeighborsClassifier":0,"LogisticRegression":0,"SVC":0,}

for i in model:
    acs = i["accuracies"]
    for a in acs:
        if (acs[a][0] == i["max"]):
            accuracies[a] = accuracies[a] + 1 
print("accuracies=" + str(accuracies))


# WHICH MODEL WITH HIGHEST SCORE
#for i in model:
#    acs = i["accuracies"]
#    for a in acs:
#        if (acs[a][0]== max_of_model):
#            print("highest accuracy score=" + str(a))
#            print("highest accuracy score model=" + str(i))


plt.hist(max_of_model)
plt.show()


plt.plot(parameters, 'o')
plt.ylabel('parameters')
plt.show()


keys = accuracies.keys()
values = accuracies.values()
plt.bar(keys, values)
plt.show()

