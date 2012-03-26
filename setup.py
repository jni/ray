from ray import imio, agglo, morpho, classify
from ray import classify2Dgeometry as c2dg
from numpy import inf
import numpy as np

def dosetup(fm = classify.MomentsFeatureManager()):
	prob = imio.read_h5_stack('im/prob5_.h5')
	#label_field = morpho.watershed(prob, seeds=(prob == 0))
	label_field = imio.read_h5_stack('im/label5_.h5')
	gs = imio.read_h5_stack('im/gs5.h5')
	cfm = c2dg.ConstellationFeatureManager(label_field.shape)
	cffm = classify.CompositeFeatureManager(children=[fm,cfm])
	return prob, label_field, gs, cfm, cffm

def doAgglo(prob, lf, gs, fm):
	print "creating train rag"
	g = agglo.Rag(lf, prob, feature_manager=fm)
	print "calculating training data"
	#g.agglomerate(inf)
	td, atd = g.learn_agglomerate(gs, fm)
	ft, l, w, h = td
	rf = classify.RandomForest()
	print "fitting RandomForest"
	rf = rf.fit(ft, l[:,0])
	lpf = agglo.classifier_probability(fm, rf)
	
	print "reading test data"
	testprob = imio.read_h5_stack('im/testprob5_.h5')
	#testlf = morpho.watershed(testprob, seeds=(testprob == 0))
	testlf = imio.read_h5_stack('im/testlabel5_.h5')
	
	print "creating test rag"
	gtest = agglo.Rag(testlf, testprob, lpf, feature_manager=fm)	
	gtest.agglomerate(inf)	
	
	
	return g, gtest, testlf
	
	
def makeAndSaveLabel(prob, filename, dtype=np.uint16):
	#Assume prob is already zero-interleaved	
	label = np.zeros(prob.shape, dtype)	
	imin = 0
	for i in range(0, prob.shape[2], 2):
		ws = morpho.watershed(prob[:,:,i])
		ws[ws > 0] = ws[ws > 0] + imin
		imin = imin + np.max(ws)
		label[:,:,i] = ws
	imio.write_h5_stack(label, filename)
	return label
		
		
	
def go(fmName, doComposite = True):	
	if fmName == "moments":
		fm = classify.MomentsFeatureManager()
	elif fmName == "histogram":
		fm = classify.HistogramFeatureManager()
	elif fmName == "convex":
		fm = classify.ConvexHullFeatureManager()
	elif fmName == "orientation":
		fm = classify.OrientationFeatureManager()
	else:
		"Unrecognized name"
		return None, None
	prob, lf, gs, cfm, cffm = dosetup(fm)
	
	if doComposite:
		g_fm, g_fm_test, lf_test = doAgglo(prob, lf, gs, cffm)
		imio.write_h5_stack(g_fm.get_ucm(), 'im/' + fmName + '_composite_ucm.h5');
		imio.write_h5_stack(g_fm.get_ucm(), 'im/' + fmName + '_composite_ucm_test.h5');
	else:
		g_fm, g_fm_test, lf_test = doAgglo(prob, lf, gs, fm)
		imio.write_h5_stack(g_fm.get_ucm(), 'im/' + fmName + '_ucm.h5');
		imio.write_h5_stack(g_fm.get_ucm(), 'im/' + fmName + '_ucm_test.h5');
	
	return g_fm, g_fm_test, lf, lf_test
