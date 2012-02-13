#!/usr/bin/env python

# system modules
import sys, os, argparse
import cPickle
import logging
from math import sqrt
from abc import ABCMeta, abstractmethod

# libraries
import h5py
import time
import itertools
import numpy as np
from numpy import bool, array, double, zeros, mean, random, concatenate, where,\
    uint8, ones, float32, uint32, unique, newaxis, zeros_like, arange, floor, \
    histogram, seterr, __version__ as numpy_version, unravel_index, diff, \
    nonzero, sort, log, inf, argsort, repeat, ones_like, cov, arccos, dot, \
    pi, bincount, isfinite, mean, median
from numpy.matlib import repmat
from numpy.linalg import det, eig, norm
from scipy import arange
from scipy.misc.common import factorial
from scipy.ndimage import binary_erosion

#from IPython.Debugger import Tracer
#debug_here = Tracer()

# local imports
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack
from adaboost import AdaBoost
from classify import NullFeatureManager

class QuickProfiler(object):
	ticd = None
	tcnt = None
	tprof = None
	logHandle = None
	logTic = 0
	logInterval = None
	
	def __init__(self, logFile = None, interval = 15000):
		if logFile is None:
			logFile = 'pyprof.txt'
		self.logHandle = open(logFile, 'w')
		self.ticd = dict()
		self.tcnt = dict()
		self.tprof = dict()
		self.logInterval = interval
		return
	
	def getMillis(self):
		t = time.time()
		return t * 1000
	
	def tic(self, strid):		
		t = self.getMillis()
		self.ticd[strid] = t
		if self.logTic == 0:
			self.logTic = t
		return
		
	def toc(self, strid):
		t = self.getMillis()
		if strid in self.tcnt.keys():
			self.tcnt[strid] += 1
			self.tprof[strid] += t - self.ticd[strid]
		else:
			self.tcnt[strid] = 1
			self.tprof[strid] = t - self.ticd[strid]
		if t - self.logTic > self.logInterval:
			self.log()
			self.logTic = 0
			#print "Logged"
		#else:
		#	print "Time since last log: " + str(t - self.logTic)
		#	print "Log Interval: " + str(self.logInterval)
		
		return
		
	def log(self):
		logstr = "{keyid} for {ms} milliseconds after {cnt} trials"
		for key in self.tprof.keys():
			self.logHandle.write(logstr.format(keyid = key, ms = self.tprof[key], cnt = self.tcnt[key]))
			print logstr.format(keyid = key, ms = self.tprof[key], cnt = self.tcnt[key])
		print ""
		self.logHandle.write("\n")
		return

	
class ConstellationFeatureManager(NullFeatureManager):
	# Pre-define some instance variables
	grids_r = None
	data_shape = None	
	profiler = None
	
	def calculate_constellation(self, g, n, ctr):
		self.profiler.tic("constellation")
		if ctr is None:
			ctr = g.node[n][self.default_cache]
		
		# nbd_idx indexes all nodes that share an edge with this one (i hope)	
		nbd_idx = g[n].keys()
		# init some arrays
		nbd_offset = zeros(len(g[n].keys()), 2)
		nbd_size = zeros(len(g[n].keys()), 1)
		
		# collect centroid offsets and sizes
		for i in arange(0,len(nbd_idx)):
			neighbor = nbd_idx[i]
			n_ctr = g.node[neighbor][self.default_cache]
			nbd_offset[i,:] = ctr - n_ctr
			nbd_size[i] = len(g.node[neighbor]['extent'])
		# sort by order around this node
		atans = np.arctan2(nbd_offset[:,0], nbd_offset[:,1])
		i_sort_tan = argsort(atans)
		nbd_offset = nbd_offset[i_sort_tan,:]
		nbd_size = nbd_size[i_sort_tan]
			
		# We want the first <consize> elements with which to make the constellation
		if len(nbd_idx) >= self.consize:
			# if there are more neighbors than consize, we want the largest <consize> of them
			i_sort_sz = argsort(nbd_size)
			# preserve the order
			i_sort_sz = sort(i_sort_sz[0:self.consize])
			nbd_offset = nbd_offset[i_sort_sz,:]
		else:
			pad = zeros((self.consize - len(nbd_idx), 2))
			nbd_offset = concatenate((ndb_offset, pad))
		
		self.profiler.toc("constellation")	
			
		return nbd_offset
	
	def z_extent(self, idxs):
		self.profiler.tic("zextent")
		#z_array = array([])		
		'''for i in idxs:
			#try:
			ix = np.remainder(i, np.prod(self.data_shape))
			xyz = unravel_index(ix, self.data_shape)
			z_array = np.append(z_array, xyz[2])
			#except ValueError as ve:
			#	debug_here()'''
		idxs = np.remainder(idxs, len(self.grids_r[2]))
		z_array = self.grids_r[2][idxs]
		
		rval = array([np.min(z_array), np.max(z_array)])
		self.profiler.toc("zextent")
		return rval
		
	def z_overlap(self, ze1, ze2, isRecurse = False):
		# lower first
		# Cases:
		# a)
		#  *----*
		#          @---------@
		# b)
		# *---------*
		#    @----------@		
		# or
		# c)
		#      *-----*
		#  @-------------@
		if not isRecurse:
			self.profiler.tic("zoverlap")
		if ze2[1] < ze1[1]:
			val =  self.z_overlap(ze2, ze1, True)
			if not isRecurse:
				self.profiler.toc("zoverlap")
			return val
		elif ze2[0] > ze1[1]:
			# Case a
			if not isRecurse:
				self.profiler.toc("zoverlap")
			return 0.0
		else:
			# first arg is Case b, second, Case c.
			zcnt = np.min([ze1[1] - ze2[0], ze1[1] - ze1[0]]) + 1
			norm = ze1[1] - ze1[0] + ze2[1] - ze2[0] + 1
			if not isRecurse:
				self.profiler.toc("zoverlap")				
			return float(zcnt) / float(norm)

	def __makeMeshGrids(self):
		shape = self.data_shape
		#xGrid, yGrid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
		#self.xGrid_r = xGrid.ravel()
		#self.yGrid_r = yGrid.ravel()
		ndim = len(shape)
		self.grids_r = dict()
		for i in np.arange(ndim):
			v = np.arange(shape[i])
			# Reshape size, should be 1, 1, ... di ... 1, 1, where d is the element in 
			# shape at index i
			svect = np.ones_like(shape)
			svect[i] = shape[i]
			# Repmat size. Should be d1, d2, ... di-1, 1, di+1, ... dn-1, dn
			# After doing a repmat, the shape of v should be equal to shape
			rvect = np.copy(shape)
			rvect[i] = 1			
			v = np.reshape(v,svect)
			v = np.tile(v,rvect)
			v = v.ravel()
			self.grids_r[i] = v
		return None

	def __init__(self, inShape, insize = 7, *args, **kwargs):
		super(ConstellationFeatureManager, self).__init__()
		self.consize = insize		
		shape = inShape
		self.data_shape = inShape
		self.__makeMeshGrids()
		self.constellation_diff = self.constellation_meansq
		self.profiler = QuickProfiler("ConstellationProfile.txt")
		print "Re-init 1"
		
	def __len__(self):
		return 1

	def constellation_meansq(self, con1, con2):
		return mean(con1 * con1 - con2 * con2)
        
	def calculate_centroid(self, node_idxs):
		idxs = np.remainder(node_idxs, len(self.grids_r[0]))
		x = mean(self.grids_r[0][idxs])
		y = mean(self.grids_r[1][idxs])
		return x, y

	def create_node_cache(self, g, n):
		self.profiler.tic("createnode")
		node_idxs = list(g.node[n]['extent'])
		ret =  np.append(self.calculate_centroid(node_idxs), self.z_extent(node_idxs))
		self.profiler.toc("createnode")
		return ret

	def create_edge_cache(self, g, n1, n2):
		self.profiler.tic("createedge")
		node1_idxs = list(g.node[n1]['extent'])
		node2_idxs = list(g.node[n2]['extent'])
		x1, y1 = self.calculate_centroid(node1_idxs)
		x2, y2 = self.calculate_centroid(node2_idxs)
		x = x2 - x1
		y = y2 - y1
        
		zext1 = self.z_extent(node1_idxs)
		zext2 = self.z_extent(node2_idxs)
		zovlp = self.z_overlap(zext1, zext2)
		self.profiler.toc("createedge")
		return x, y, zovlp

	# Super way not efficient, but simpler for testing this thing
	# TODO: use existing cache to calculate new centroid
	def update_node_cache(self, g, n1, n2, dst, src):
		dst = self.create_node_cache(g,n1)

	def update_edge_cache(self, g, e1, e2, dst, src):
		dst = self.create_edge_cache(g, e1[0], e1[1])

	def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
		pass

	def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
		pass

	def compute_node_features(self, g, n, cache=None):
		if cache is None: 
			cache = g.node[n][self.default_cache]		
		return self.calculate_constellation(g, n, cache).ravel()

	def compute_edge_features(self, g, n1, n2, cache=None):		
		if cache is None:
			cache = g[n1][n2][self.default_cache]		
		return cache[2]

	def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
		if cache1 is None:
			cache1 = g.node[n1][self.default_cache]		

		if cache2 is None:
			cache2 = g.node[n2][self.default_cache]		
		
		con1 = self.compute_node_features(g, n1, cache1)
		con2 = self.compute_node_features(g, n2, cache2)
		
		return self.constellation_diff(con1, con2)
