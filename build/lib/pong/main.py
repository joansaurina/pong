#!/usr/bin/env python3
# Ensure env in case this file is being run as an executable

'''
author: Aaron Behr
created: 2014-06-29
'''
import sys
import numpy as np
import os
from os import path
import argparse
import time
import json
from shutil import rmtree
import tornado.ioloop
import tornado.web
import tornado.websocket
import munkres # not used here, just checking version
import networkx as nx # not used here, just checking version

#pylint: enable=import-error

with open(path.join(path.dirname(__file__), 'VERSION')) as f:
	version = f.read().strip()

year = 2021
sys.path.insert(0, path.dirname(__file__))
import parse, cm, write, align, distruct

clients = []
threads = []
pongdata = None
run_pong_args = None

class Pongdata:
	def __init__(self, intro, outputdir, printall):
		self.runs = {} # contains all Run objects
		self.all_kgroups = [] # contains kgroups in order
		self.cluster_matches = {} # all clustering solutions matching 2 runs

		self.name2id = {} # run name to run ID


		self.num_indiv = -1
		self.K_min = -1
		self.K_max = -1

		self.intro = intro
		self.output_dir = outputdir
		self.print_all = printall

		self.ind2pop = None
		self.pop_order = None
		self.popcode2popname = None
		self.popindex2popname = None
		self.pop_sizes = None
		self.sort_by = None
		self.indiv_avg = None

		self.colors = [] # use custom colors?

		# status attr is only necessary if pong is run from within the server
		# self.status = 0 # incomplete, working, or complete (0,1,2)

intro = '\n'
intro += '-------------------------------------------------------------------\n'
intro += '                            p o n g\n'
intro += '      by A. Behr, K. Liu, T. Devlin, G. Liu-Fang, and S. Ramachandran\n'
intro += '                       Version %s (%d)\n' % (version, year)
intro += '-------------------------------------------------------------------\n'
intro += '-------------------------------------------------------------------\n'






def main():
	dist_metrics = ['sum_squared', 'percent', 'G', 'jaccard']
	
	parser = argparse.ArgumentParser(description='-------------------------------- '
		'pong, v%s --------------------------------' % version)

	parser.add_argument('-m', '--filemap', required=True,
		help='path to params file containing information about input '
		'Q-matrix files')
	parser.add_argument('-c', '--ignore_cols', type=int, default = 0,
		help='ignore the first i columns of every data line. Typically 5 for '
		'Structure output and 0 for ADMIXTURE output. Default = 0')
	parser.add_argument('-o', '--output_dir', default=None, # gets set later
		help='specify output dir for files to be '
		'written to. By default, pong makes a folder named "pong_output_datetime" in '
		'the current working directory, where "datetime" is the current system date and time.')

	parser.add_argument('-i', '--ind2pop', default=None,
		help='ind2pop data (can be either a Q-matrix column number or the ' 
		'path to a file containing the data).')
	parser.add_argument('-n', '--pop_names', default=None,
		help='Path to file containing population order/names.')
	parser.add_argument('-l', '--color_list',
		help='List of colors to be used for visualization. If this file is not '
		'included, then default colors will be used for visualization.')
	parser.add_argument('-f', '--force', default=False,
		action='store_true', help='force overwrite already existing output '
		'directory. By default, pong will prompt the user before overwriting.')

	parser.add_argument('-s', '--sim_threshold', type=float,
		default=0.97, help='choose threshold to combine redundant clusters at '
		'a given K. Default = 0.97')
	parser.add_argument('--col_delim', default=None,
		help='Provide the character on which to split columns. Default is '
		'whitespace (of any length).')
	parser.add_argument('--dist_metric',
		default='jaccard', help='distance metric to be used for comparing '
		'cluster similarities. Choose from %s. Default = jaccard' 
		% str(dist_metrics))
	parser.add_argument('--disable_server', default=False, action='store_true',
		help='run pong\'s algorithm without initializing a server instance or '
		'visualizing results.')
	parser.add_argument('-p', '--port', type=int, default=4000,
		help='Specify port on which the server should locally host. Default = 4000.')
	parser.add_argument('-v', '--verbose', default=False,
		action='store_true', help='Report more details about clustering '
		'results to the command line, and print all cluster distances in the '
		'output files (by default, only the best 5 are printed).')

	parser.add_argument('-g', '--greedy', default=False, action='store_true',
		help='Force the use of the greedy algorithm if a set of disjoint '
		'cliques cannot be found. By default, pong prompts the user with a '
		'choice of whether to continue with the greedy algorithm, or to '
		'exit and re-run with different parameters.')
	opts = parser.parse_args()

	# Check system Python version and dependency versions. These are enforced
	# when installing/upgrading via pip, but not if running dev version.
	if sys.version_info.major != 3:
		sys.exit('Error: You are running Python %d; pong requires version 3.' % sys.version_info.major)
	
	fmt_v = lambda module: module.__version__.split('.')
	deps = True # dependencies are good
	deps = deps and sys.version_info.minor >= 6 # python 3.7 or higher
	deps = deps and int(fmt_v(np)[0]) == 1 # numpy v1
	deps = deps and int(fmt_v(np)[1]) >= 19 # 1.19 or higher
	deps = deps and int(fmt_v(munkres)[0]) == 1 # munkres v1
	deps = deps and int(fmt_v(munkres)[1]) >= 1 # 1.1 or higher
	deps = deps and int(fmt_v(nx)[0]) == 2 # networkx v2
	deps = deps and int(fmt_v(nx)[1]) >= 5 # 2.5 or higher
	deps = deps and int(tornado.version_info[0]) == 6 # tornado v6

	if not deps:
		sys.stdout.write(f'Warning: pong expects the following dependencies:\n'
			f' - python >= 3.7 (installed: v3.{sys.version_info.minor}),\n'
			f' - numpy >= 1.18 (installed: {np.__version__}),\n'
			f' - munkres >= 1.1 (installed: {munkres.__version__}),\n'
			f' - networkx >= 2.4 (installed: {nx.__version__}),\n'
			f' - tornado >= 6 (installed: {tornado.version}).\n'
			f'We recommend upgrading these modules, otherwise you may\n'
			f'experience issues running pong.\n')
		r = input('Continue anyway? (y/n): ')
		while r not in ('y', 'Y', 'n', 'N'):
			r = input('Please enter "y" to overwrite or "n" to exit: ')
		if r in ('n', 'N'): sys.exit(1)


	# Check validity of pongparams file
	pong_filemap = path.abspath(opts.filemap)
	if not path.isfile(pong_filemap):
		sys.exit('Error: Could not find pong filemap at %s.' % pong_filemap)

	# Check validity of specified distance metric
	if not opts.dist_metric in dist_metrics:
		x = (opts.dist_metric, str(dist_metrics))
		sys.exit('Invalid distance metric: "%s". Please choose from %s' % x)

	printall = opts.verbose
	
	ind2pop = None
	labels = None

	if opts.ind2pop is not None:
		try:
			ind2pop = int(opts.ind2pop)
		except ValueError:
			ind2pop = path.abspath(opts.ind2pop)
			if not path.isfile(ind2pop):
				sys.exit('Error: Could not find ind2pop file at %s.' % ind2pop)
	

	if opts.pop_names is not None:
		if ind2pop is None:
			sys.exit('Error: must provide ind to pop data in order to provide '
				'pop order data')
		labels = path.abspath(opts.pop_names)
		if not path.isfile(labels):
			sys.exit('Error: Could not find pop order file at %s.' % labels)




	# Check validity of color file
	colors = []
	color_file = opts.color_list
	if color_file:
		color_file = path.abspath(color_file)
		if not path.isfile(color_file):
			sys.stdout.write('\nWarning: Could not find color file '
				'at %s.\n' % color_file)
			
			r = input('Continue using default colors? (y/n): ')
			while r not in ('y', 'Y', 'n', 'N'):
				r = input('Please enter "y" to overwrite or '
					'"n" to exit: ')
			if r in ('n', 'N'): sys.exit(1)

			color_file = None
		else:
			sys.stdout.write('\nCustom colors provided. Visualization utilizes the '
				'color white.\nIf color file contains white, users are advised to '
				'replace it with another color.\n')
			with open(color_file, 'r') as f:
				colors = [x for x in [l.strip() for l in f] if x != '']


	# Check and clean output dir
	outputdir = opts.output_dir
	if outputdir:
		outputdir = path.abspath(outputdir)
	else:
		dirname = 'pong_output_' + time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
		outputdir = path.abspath(path.join(os.getcwd(), dirname))
	
	if os.path.isdir(outputdir):
		if opts.force:
			rmtree(outputdir)
		else:
			outputdir_name = os.path.split(outputdir)[1]
			print('\nOutput dir %s already exists.' % outputdir_name)

			r = input('Overwrite? (y/n): ')
			while r not in ('y', 'Y', 'n', 'N'):
				r = input('Please enter "y" to overwrite or "n" to exit: ')
			if r in ('n', 'N'): sys.exit(1)
			rmtree(outputdir)

	os.makedirs(outputdir)


	# Initialize object to hold references to all main pong data
	global pongdata
	pongdata = Pongdata(intro, outputdir, printall)
	pongdata.colors = colors

	params_used = intro+'\n\n' # ===============\n
	params_used += 'pong_filemap file: %s\n' % pong_filemap
	params_used += 'Distance metric: %s\n' % opts.dist_metric
	params_used += 'Similarity threshold: %f\n' % opts.sim_threshold
	params_used += 'Verbose: %s\n' % str(pongdata.print_all)
	params_used += '\nFull command: ' + ' '.join(sys.argv[:]) + '\n'

	pongdata.sim_threshold = opts.sim_threshold

	with open(os.path.join(pongdata.output_dir, 'params_used.txt'), 'w') as f:
		f.write(params_used)


	global run_pong_args
	run_pong_args = (pongdata, opts, pong_filemap, labels, ind2pop)


	# ========================= RUN PONG ======================================

	print(pongdata.intro)


	# Code for running pong from the tornado app
	# if opts.disable_server:
	# 	run_pong(*run_pong_args)
	# else:
	# 	app = Application()
	# 	app.listen(opts.port)

	# 	msg = 'pong server is now running locally & listening on port %s\n' % opts.port
	# 	msg += 'Open your web browser and navigate to localhost:%s to see the visualization\n\n'% opts.port
	# 	sys.stdout.write(msg)
		
	# 	try:
	# 		tornado.ioloop.IOLoop.current().start()
	# 	except KeyboardInterrupt:
	# 		sys.stdout.write('\n')
	# 		sys.exit(0)


	run_pong(*run_pong_args)

	# Always save SVG directly
	output_svg_path = path.join(pongdata.output_dir, 'visualization.svg')
	print('Generating SVG visualization...')
	generate_svg_from_data(pongdata, output_svg_path)
	print(f'SVG saved to: {output_svg_path}')
	print('\nVisualization complete. SVG file saved in output directory.')





def generate_svg_from_data(pongdata, output_filename):
	"""
	Generate SVG visualization directly from pongdata without using a browser.
	"""
	runs = pongdata.runs
	all_kgroups = pongdata.all_kgroups
	
	# Default colors
	colors = ["#E04B4B", "#6094C3", "#63BC6A", "#A76BB2", "#F0934E",
		"#FEFB54", "#B37855", "#EF91CA", "#A4A4A4"]
	colors_26 = ["#f0a3ff", "#0075dc", "#993f00", "#4c005c", "#191919", 
		"#005c31", "#2bce48", "#ffcc99", "#808080", "#94ffb5", "#8f7c00", 
		"#9dcc00", "#c20088", "#003380", "#ffa405", "#ffa8bb", "#426600", 
		"#ff0010", "#5ef1f2", "#00998f", "#e0ff66", "#740aff", "#990000", 
		"#ffff80", "#ffff00", "#ff5005"]
	
	if len(pongdata.colors) > 0:
		colors = pongdata.colors
	elif pongdata.K_max > 9:
		colors = colors_26
	
	# SVG dimensions
	svg_width = 1200
	plot_width = svg_width * 0.84
	translate_x = svg_width * 0.08
	plot_height = 160 * 0.8
	svg_height_per_plot = 175
	
	svg_elements = []
	
	# Generate SVG for each K value
	for kgroup in all_kgroups:
		K = kgroup.K
		primary_run = runs[kgroup.primary_run]
		color_perm = kgroup.color_perm
		
		# Get population data
		if not hasattr(primary_run, 'population_object_data') or primary_run.population_object_data is None:
			continue
		
		pop_data = primary_run.population_object_data
		num_individuals = sum(len(pop['members']) for pop in pop_data)
		
		# Create SVG for this K
		svg_parts = []
		svg_parts.append(f'<svg width="{svg_width}" height="{svg_height_per_plot}" xmlns="http://www.w3.org/2000/svg">')
		
		# Add K label
		svg_parts.append(f'<text x="0" y="{plot_height/2 + 10}" font-family="Helvetica" font-size="20" font-weight="bold">K = {K}</text>')
		
		# Create chart group
		svg_parts.append(f'<g transform="translate({translate_x}, 15)">')
		
		# Calculate x scale
		x_max = num_individuals
		x_scale = plot_width / x_max if x_max > 0 else plot_width
		
		# Add background rectangle first (so everything else appears on top)
		svg_parts.append(f'<rect x="-2.5" y="-2.5" width="{plot_width + 5}" height="{plot_height + 5}" fill="black"/>')
		
		# Draw stacked areas for each population
		current_x = 0
		tick_positions = []  # Store positions for vertical separator lines
		for pop_idx, pop in enumerate(pop_data):
			pop_size = len(pop['members'])
			# Store position of first member of each population for vertical lines
			if pop['members']:
				first_member_idx = pop['members'][0]['index']
				tick_x = first_member_idx * x_scale
				tick_positions.append(tick_x)
			
			# Collect all member data with their cluster values
			members_data = []
			for member in pop['members']:
				idx = member['index']
				cluster_vals = {}
				for key, value in member.items():
					if key.startswith('cluster'):
						cluster_num = int(key.replace('cluster', '')) - 1
						cluster_vals[cluster_num] = value
				members_data.append((idx, cluster_vals))
			
			# Sort by index
			members_data.sort(key=lambda x: x[0])
			
			# Draw each cluster as a stacked area (from bottom to top)
			# Using step-after curve like D3
			for cluster_idx in range(K):
				color_idx = color_perm[cluster_idx] if cluster_idx < len(color_perm) else cluster_idx
				color = colors[color_idx % len(colors)]
				
				# Build path for this cluster layer using step-after
				path_points = []
				
				# Bottom edge (going left to right) - step after
				for i, (idx, cluster_vals) in enumerate(members_data):
					# Calculate cumulative Y from bottom
					cumulative_bottom = sum(cluster_vals.get(c, 0) for c in range(cluster_idx))
					cumulative_top = cumulative_bottom + cluster_vals.get(cluster_idx, 0)
					
					y_bottom = plot_height - (cumulative_bottom * plot_height)
					y_top = plot_height - (cumulative_top * plot_height)
					
					if i == 0:  # First point
						path_points.append(f'M {idx * x_scale} {y_bottom}')
					else:
						# Step after: horizontal line first, then vertical
						prev_idx = members_data[i-1][0]
						path_points.append(f'H {idx * x_scale}')
					
					path_points.append(f'V {y_top}')
				
				# Add final horizontal step
				if members_data:
					last_idx = members_data[-1][0]
					path_points.append(f'H {(last_idx + 1) * x_scale}')
				
				# Top edge (going right to left) - step after
				for i in range(len(members_data) - 1, -1, -1):
					idx, cluster_vals = members_data[i]
					cumulative_bottom = sum(cluster_vals.get(c, 0) for c in range(cluster_idx))
					y_bottom = plot_height - (cumulative_bottom * plot_height)
					
					if i == len(members_data) - 1:
						path_points.append(f'V {y_bottom}')
					else:
						path_points.append(f'V {y_bottom}')
						next_idx = members_data[i+1][0]
						path_points.append(f'H {idx * x_scale}')
				
				# Close to start
				if members_data:
					first_idx = members_data[0][0]
					path_points.append(f'H {first_idx * x_scale}')
				
				path_points.append('Z')
				
				svg_parts.append(f'<path d="{" ".join(path_points)}" fill="{color}" stroke="none"/>')
			
			current_x += pop_size
		
		# Draw horizontal line at top (y=0) - after areas so it's visible on top
		svg_parts.append(f'<line x1="0" y1="0" x2="{plot_width}" y2="0" stroke="#000000" stroke-width="1" stroke-opacity="1"/>')
		
		# Draw vertical separator lines at the start of each population (except first)
		# These lines go from top (y=0) to bottom (y=plot_height) - like tickSize(-indivHeight) in D3
		# Draw lines for all populations except the first one
		if len(tick_positions) > 1:
			for tick_x in tick_positions[1:]:  # Skip first position (no line needed at start)
				# Draw line with explicit coordinates - ensure it's drawn after all paths
				# Use stroke="white" temporarily to test visibility, then change back to "black"
				svg_parts.append(f'<line x1="{tick_x}" y1="0" x2="{tick_x}" y2="{plot_height}" stroke="black" stroke-width="1" stroke-opacity="1"/>')
		
		svg_parts.append('</g>')  # Close chart group
		svg_parts.append('</svg>')
		
		svg_elements.append({
			'content': '\n'.join(svg_parts),
			'height': svg_height_per_plot
		})
	
	# Combine all SVGs
	if not svg_elements:
		sys.exit('Error: No visualization data available to generate SVG.')
	
	max_width = svg_width
	total_height = sum(elem['height'] + 20 for elem in svg_elements) - 20
	
	# Create final SVG
	svg_content = []
	svg_content.append('<?xml version="1.0" encoding="UTF-8"?>')
	svg_content.append(f'<svg width="{max_width}" height="{total_height}" xmlns="http://www.w3.org/2000/svg">')
	
	current_y = 0
	for svg_elem in svg_elements:
		# Extract content from inner SVG
		inner_svg = svg_elem['content']
		# Remove outer <svg> tags and add as group
		inner_content = inner_svg.replace('<svg width="1200" height="175" xmlns="http://www.w3.org/2000/svg">', '').replace('</svg>', '')
		svg_content.append(f'<g transform="translate(0, {current_y})">')
		svg_content.append(inner_content)
		svg_content.append('</g>')
		current_y += svg_elem['height'] + 20
	
	svg_content.append('</svg>')
	
	# Write to file
	output_path = path.abspath(output_filename)
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(svg_content))


def run_pong(pongdata, opts, pong_filemap, labels, ind2pop):
	pongdata.status = 1

	t0=time.time()
	# PARSE INPUT FILE AND ORGANIZE DATA INTO GROUPS OF RUNS PER K
	print('Parsing input and generating cluster network graph')
	parse.parse_multicluster_input(pongdata, pong_filemap, opts.ignore_cols, 
		opts.col_delim, labels, ind2pop)


	# MATCH CLUSTERS FOR RUNS WITHIN EACH K AND CONDENSE TO REPRESENTATIVE RUNS
	print('Matching clusters within each K and finding representative runs')
	t1 = time.time()
	cm.clump(pongdata, opts.dist_metric, opts.sim_threshold, opts.greedy)

	# MATCH CLUSTERS ACROSS K
	print('Matching clusters across K')
	cm.multicluster_match(pongdata, opts.dist_metric)
	t2 = time.time()

	# PRINT MATCH CLUSTERS RESULTS
	write.output_cluster_match_details(pongdata)
	
	# print(pongdata.name2id)
	# COMPUTE BEST-GUESS ALIGNMENTS FOR ALL RUNS WITHIN AND ACROSS K
	print('Finding best alignment for all runs within and across K')
	t3 = time.time()
	align.compute_alignments(pongdata, opts.sim_threshold)
	t4 = time.time()

	if pongdata.print_all:
		# PRINT BEST-FIT ALIGNMENTS
		write.output_alignments(pongdata)


	# GENERATE COLOR INFO
	parse.convert_data(pongdata)
	distruct.generate_color_perms(pongdata)
	if len(pongdata.colors) > 0:
		if (pongdata.print_all):
			print('Generating perm files for Distruct')
			distruct.generate_distruct_perm_files(pongdata, pongdata.colors)
	

	pongdata.status = 2
	
	# write.write_json(pongdata)

	print('match time: %.2fs' % (t2-t1))
	print('align time: %.2fs' % (t4-t3))
	print('total time: %.2fs' % ((t2-t0)+(t4-t3)))






class Application(tornado.web.Application):
	def __init__(self):
		handlers = [
			(r"/", MainHandler),
			(r"/pongsocket", WSHandler),
		]
		src = path.dirname(__file__) # if version == 'DEV' else pong.__path__[0]
		settings = dict(
			template_path=path.join(src, "templates"),
			static_path=path.join(src, "static"),
		)
		tornado.web.Application.__init__(self, handlers, **settings)


class MainHandler(tornado.web.RequestHandler):
	def get(self):
		self.render("pong.html")

class WSHandler(tornado.websocket.WebSocketHandler):
	global pongdata
	clients = set()

	def open(self):
		WSHandler.clients.add(self)
		
		# Code for running pong from the tornado app
		# Server is not asynchronous so it won't serve a partially-completed Pong object
		# if pongdata.status == 0:
			# global run_pong_args
			# run_pong(*run_pong_args)
		
		print('New browser connection; generating visualization')
		pong_json_data = write.write_json(pongdata) # add 'True' when debugging to get json

		self.write_message(json.dumps({'type': 'pong-data',
			'pong': pong_json_data},))



	def on_close(self):
		WSHandler.clients.remove(self)
		print('Browser disconnected')

	# @classmethod
	# def update(cls, data):
	#	 for client in cls.clients:
	#		 client.write_message(data)

	def on_message(self, message):
		# logging.info("received message")

		data = json.loads(message)
		data = tornado.escape.json_decode(message)

		#received call from client on_message getQmatrix function call
		if data['type'] == 'get-qmatrix':
			name = data['name']
			run = pongdata.runs[pongdata.name2id[name]] #returns run instance
			minor = data['minor']
			minorID = data['minorID']
			is_first = data['is_first']

			# print 'server received request for Q-matrix %s. Column perm %s.' % (name, str(run.alignment-1))

			if minor=='yes':
				response = {'type':'q-matrix', 'name':name, 'K':run.K,'matrix2d':run.population_object_data, 'minor':'yes', 'minorID':minorID, 'is_first':is_first}
			else:
				response = {'type':'q-matrix', 'name':name, 'K':run.K, 'matrix2d':run.population_object_data, 'minor':'no', 'minorID': None, 'is_first':None}

			self.write_message(json.dumps(response))

		else:
			sys.exit('Error: Received invalid socket message from client')





if __name__ == '__main__':
	main()
