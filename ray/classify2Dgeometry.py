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
from numpy.linalg import det, eig, norm
from scipy import arange
from scipy.misc.common import factorial
from scipy.ndimage import binary_erosion

from IPython.Debugger import Tracer
debug_here = Tracer()

# local imports
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack
from adaboost import AdaBoost
from classify import NullFeatureManager

class ConstellationFeatureManager(NullFeatureManager):
	
	# Pre-define some instance variables
	xGrid_r = None
	yGrid_r = None
	data_shape = None	
	
	def calculate_constellation(self, g, n, ctr):
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
			
		return nbd_offset
	
	def z_extent(self, idxs):
		z_array = array([])		
		for i in idxs:
			try:
				ix = np.remainder(i, np.prod(self.data_shape))
				xyz = unravel_index(ix, self.data_shape)
				z_array = np.append(z_array, xyz[2])
			except ValueError as ve:
				debug_here()
		
		rval = array([np.min(z_array), np.max(z_array)])
		return rval
		
	def z_overlap(self, ze1, ze2):
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
		if ze2[1] <= ze1[1]:
			return self.z_overlap(ze2, ze1)
		elif ze2[0] > ze1[1]:
			# Case a
			return 0.0
		else:
			# first arg is Case b, second, Case c.
			zcnt = np.min([ze1[1] - ze2[0], ze1[1] - ze1[0]]) + 1
			norm = ze1[1] - ze1[0] + ze2[1] - ze2[0] + 1
			return float(zcnt) / float(norm)

	def __makeMeshGrids(self):
		shape = self.data_shape
		xGrid, yGrid = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
		self.xGrid_r = xGrid.ravel()
		self.yGrid_r = yGrid.ravel()
		return None

	def __init__(self, inShape, insize = 7, *args, **kwargs):
		super(ConstellationFeatureManager, self).__init__()
		self.consize = insize		
		shape = inShape
		self.data_shape = inShape
		self.__makeMeshGrids()
		self.constellation_diff = self.constellation_meansq
		
	def __len__(self):
		return 1

	def constellation_meansq(self, con1, con2):
		return mean(con1 * con1 - con2 * con2)
        
	def calculate_centroid(self, node_idxs):
		idxs = np.remainder(node_idxs, len(self.xGrid_r))
		x = mean(self.xGrid_r[idxs])
		y = mean(self.yGrid_r[idxs])
		return x, y

	def create_node_cache(self, g, n):		
		node_idxs = list(g.node[n]['extent'])
		return np.append(self.calculate_centroid(node_idxs), self.z_extent(node_idxs))

	def create_edge_cache(self, g, n1, n2):                
		node1_idxs = list(g.node[n1]['extent'])
		node2_idxs = list(g.node[n2]['extent'])
		x1, y1 = self.calculate_centroid(node1_idxs)
		x2, y2 = self.calculate_centroid(node2_idxs)
		x = x2 - x1
		y = y2 - y1
        
		zext1 = self.z_extent(node1_idxs)
		zext2 = self.z_extent(node2_idxs)
		zovlp = self.z_overlap(zext1, zext2)

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
