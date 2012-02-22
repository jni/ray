from ray import imio, agglo, morpho, classify
from ray import classify2Dgeometry as c2dg
from numpy import inf

def dosetup(fm = classify.MomentsFeatureManager()):
	prob = imio.read_h5_stack('im/prob5.h5')
	label_field = imio.read_h5_stack('im/label5.h5')
	gs = imio.read_h5_stack('im/gs5.h5')
	cfm = c2dg.ConstellationFeatureManager(label_field.shape)
	cffm = classify.CompositeFeatureManager(children=[fm,cfm])
	return prob, label_field, gs, cfm, cffm

def doAgglo(prob, lf, gs, fm):
	print "creating train rag"
	g = agglo.Rag(lf, prob, feature_manager=fm)
	print "calculating training data"
	td, atd = g.learn_agglomerate(gs, fm)
	ft, l, w, h = td
	rf = classify.RandomForest()
	print "fitting RandomForest"
	rf = rf.fit(ft, l[:,0])
	lpf = agglo.classifier_probability(fm, rf)
	
	print "reading test data"
	testprob = imio.read_h5_stack('im/testprob5.h5')
	testlf = imio.read_h5_stack('im/testlabel5.h5')
	
	print "creating test rag"
	#gtest = agglo.Rag(testlf, testprob, lpf)	
	
	return g, testprob, testlf, td, atd, lpf
