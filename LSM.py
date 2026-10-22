import pyvisa
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import os

def verbose_trigger_type(trigger_type):
	# bit number 	trigger type
	# 0 	 		Photodiode through DMM7510
	# 1 			hardware trigger
	if bool(trigger_type & 1):
		print("Triggering the photodiode measurement")
	else:
		print("NOT triggering the photodiode measurement")
	if bool((trigger_type>>1) & 1):
		print("Triggering the hardware trigger")
	else:
		print("NOT triggering the hardware trigger")
		
def save_image(path, name, x, y, image):
	os.makedirs(path, exist_ok=True)
	im=Image.fromarray(image)
	im.save(f'{path}/{name}.tiff')
	with open(f"{path}/{name}.log", "w") as f:
		np.savetxt(f, x, delimiter='\t', newline='\t')
		f.write('\n')
		np.savetxt(f, y, delimiter='\t', newline='\t')
	print(f"Saved {path}/{name}.tiff and {path}/{name}.log")

def rescale(output, objective=5, apply=True):
	if not apply:
		return output

	if objective not in [5,10,20,50,100]:
		print("Unknown objective")
		return output
	objectives=np.array([5,10,20,50,100])
	xfactor=[621.576367,308.087566,155.393470,62.166693,29.721644]
	yfactor=[643.970523,320.422395,163.000598,64.654161,31.549683]
	idx=np.argmin(np.abs(objectives-objective))
	x=output[0]*xfactor[idx]
	y=output[1]*yfactor[idx]
	im=output[2]

	return x,y,im

def save_line(path, name, varying, fixed, line):
	os.makedirs(path, exist_ok=True)
	with open (f"{path}/{name}.line", 'w+') as f:
		np.savetxt(f, [varying])
		np.savetxt(f,[line])
		f.write(f"# fixed  {fixed}")
	print(f"Saved {path}/{name}.line")

def centered_to_framed(xcenter, width, ycenter, height):
	xmin = xcenter - width / 2
	xmax = xcenter + width / 2
	ymin = ycenter - height / 2
	ymax = ycenter + height / 2
	return [xmin, xmax, ymin, ymax]

def framed_to_centered(xmin, xmax, ymin, ymax):
	xcenter=(xmin+xmax)/2
	ycenter=(ymin+ymax)/2
	width=xmax-xmin
	height=ymax-ymin

	return [xcenter, width, ycenter, height]

class LSM:

	def __init__(self, K_2612b_address, timeout=30e3, voltage_limit=10, dmm_nplc=0.01):
		rm=pyvisa.ResourceManager()
		try:
			self.k=rm.open_resource(K_2612b_address)
			self.k.timeout=timeout
			self.timeout=timeout
			self.nplc=dmm_nplc
			self.voltage_limit=voltage_limit
			self.k.clear()
			self.k.write('*CLS')
			self.flashed=False
		except Exception as e:
			print(e)
			print('Failed to connect to the instrument')

	def __del__(self):
		self.k.close()

	def verbose_write(self, command, verbose):
		if verbose:
			print(f"Sending : {command}")
		return self.k.write(command)

	def enforce_limits(self, xmin, xmax, ymin, ymax):
		if abs(xmin)>self.voltage_limit:
			print(f"Reducing xmin from {xmin} to ", end='')
			xmin=np.sign(xmin)*self.voltage_limit
			print(xmin)
		if abs(ymin)>self.voltage_limit:
			print(f"Reducing ymin from {ymin} to ", end='')
			ymin=np.sign(ymin)*self.voltage_limit
			print(ymin)
		if abs(xmax)>self.voltage_limit:
			print(f"Reducing xmax from {xmax} to ", end='')
			xmax=np.sign(xmax)*self.voltage_limit
			print(xmax)
		if abs(ymax)>self.voltage_limit:
			print(f"Reducing ymax from {ymax} to ", end='')
			ymax=np.sign(ymax)*self.voltage_limit
			print(ymax)
		return [xmin, xmax, ymin, ymax]

	def enforce_limits_1D(self, smin, smax):
		if abs(smin)>self.voltage_limit:
			print(f"Reducing smin from {smin} to ", end='')
			smin=np.sign(smin)*self.voltage_limit
			print(smin)
		if abs(smax)>self.voltage_limit:
			print(f"Reducing smax from {smax} to ", end='')
			smax=np.sign(smax)*self.voltage_limit
			print(smax)
		return [smin, smax]
	
	def enforce_limits_0D(self, s):
		if abs(s)>self.voltage_limit:
			print(f"Reducing s from {s} to ", end='')
			s=np.sign(s)*self.voltage_limit
			print(s)
		return s

	def flash_script(self, dmm_range=6, end_type='zero', hardware_output_trigger_pin=1, hardware_output_trigger_time=1e-4, triggers_to_initialize=[]):
		if end_type not in ["zero", "last", "first"]:
			print('end_type must be either "zero" (returns to 0), "last" (holds the last voltage) or "first" (returns to the first value)')
			print('Using "zero"')
			end_type="zero"
		if end_type=="zero":
			end_x="0"
			end_y="0"
			end_s='0'
			end_f='0'
		elif end_type=="last":
			end_x="Xmax"
			end_y="Ymax"
			end_s='Smax'
			end_f='fixed'
		elif end_type=="first":
			end_x="Xmin"
			end_y="Ymin"
			end_s='Smin'
			end_f='fixed'
		try:
			tsp_script=f"""
loadscript makeFunc
	function initiateInstrument()
		tsplink.reset()
	    smu = node[1]  -- 2612B
	    dmm = node[2]  -- DMM7510
	    dmm.reset()
	    smu.smua.reset()
	    smu.smub.reset()
	    smu.smua.source.func = smu.smua.OUTPUT_DCVOLTS
	    smu.smub.source.func = smu.smub.OUTPUT_DCVOLTS
	    smua.source.autorangev = smua.AUTORANGE_OFF
	    smub.source.autorangev = smub.AUTORANGE_OFF
	    smu.smua.source.levelv = 0
	    smu.smub.source.levelv = 0
	    smu.smua.source.output = smu.smua.OUTPUT_ON
	    smu.smub.source.output = smu.smub.OUTPUT_ON 
	    dmm.dmm.measure.func = dmm.dmm.FUNC_DC_VOLTAGE
	    dmm.dmm.measure.range = {dmm_range}
	    dmm.dmm.measure.nplc = {self.nplc}
		computer_flag=0
	    digio.writebit({hardware_output_trigger_pin}, 0)
	end

	function getFirstBit(num)
		return math.mod(num,2)
	end

	function getSecondBit(num)
		return math.floor(num/2)
	end

	function changeDMMNPLC(nplc)
		dmm.dmm.measure.nplc = nplc
	end

	function changeDMMRange(range)
		dmm.dmm.measure.range = range
	end
		

	function fillBufferManualTrigger(triggerPin, sampleCount)
		dmm.defbuffer1.capacity = sampleCount
		for j = 0, sampleCount-1 do
			softwareTrigger()
			manualTrigger(triggerPin)
		end
		printbuffer(1, dmm.defbuffer1.n, dmm.defbuffer1)
	end

	function rasterScan(Xmin, Xmax, Nx, Ymin, Ymax, Ny, settleTime, returnLineSettleTime, triggerType)
	    local dx = (Xmax - Xmin) / (Nx - 1)
	    local dy = (Ymax - Ymin) / (Ny - 1)
	    local sampleCount = Nx * Ny
	    smu.smua.source.rangev = math.max(math.abs(Xmin), math.abs(Xmax))
	    smu.smub.source.rangev = math.max(math.abs(Ymin), math.abs(Ymax))
	    if getFirstBit(triggerType) ~= 0 then
			dmm.defbuffer1.capacity = sampleCount
		end
		local st=(getFirstBit(triggerType))
		local ht=(getSecondBit(triggerType))
	    print("Starting")
	    for j = 0, Ny-1 do
	        local y = Ymin + j * dy
	        smu.smub.source.levelv = y
			smu.smua.source.levelv = Xmin
			delay(returnLineSettleTime)
	        for i = 0, Nx-1 do
	            local x = Xmin + i * dx
	            smu.smua.source.levelv = x
	            delay(settleTime)
				if st ~= 0 then 
					softwareTrigger() 
				end
            	if ht ~= 0 then 
					hardwareTrigger() 
				end
	        end
	    end
	    print(1)
	    smu.smua.source.levelv = {end_x}
	    smu.smub.source.levelv = {end_y}
		if (getFirstBit(triggerType)) ~= 0 then 
	    	printbuffer(1, dmm.defbuffer1.n, dmm.defbuffer1)
		end
	end
	function rasterScanBufferSynced(Xmin, Xmax, Nx, Ymin, Ymax, Ny, settleTime, returnLineSettleTime, triggerType, bufferSize)
	    local dx = (Xmax - Xmin) / (Nx - 1)
	    local dy = (Ymax - Ymin) / (Ny - 1)
	    local sampleCount = Nx * Ny
	    be=0
	    smu.smua.source.rangev = math.max(math.abs(Xmin), math.abs(Xmax))
	    smu.smub.source.rangev = math.max(math.abs(Ymin), math.abs(Ymax))
	    if getFirstBit(triggerType) ~= 0 then
			dmm.defbuffer1.capacity = sampleCount
		end
	    print("Starting")
	    local currentIndex=0
	    for j = 0, Ny-1 do
	        local y = Ymin + j * dy
	        smu.smub.source.levelv = y
			smu.smua.source.levelv = Xmin
			delay(returnLineSettleTime)
	        for i = 0, Nx-1 do
	            local x = Xmin + i * dx
	            smu.smua.source.levelv = x
	            delay(settleTime)
				if (getFirstBit(triggerType)) ~= 0 then 
					softwareTrigger() 
				end
            	if (getSecondBit(triggerType)) ~= 0 then 
					hardwareTrigger() 
				end
	        end
	        currentIndex=currentIndex+Nx
	        if currentIndex>bufferSize then
	        	print("waiting for computer_flag")
				local t0 = timer.tick
				local timeout_s = 30
				while computer_flag == 0 do
				    if (timer.tick - t0) > timeout_s then
				        error("PC did not acknowledge within "..timeout_s.." s")
				    end
				    delay(0.05) -- polite polling
				end
				computer_flag=0

	        end


	    end
	    smu.smua.source.levelv = {end_x}
	    smu.smub.source.levelv = {end_y}
	    print(1)
		if (getFirstBit(triggerType)) ~= 0 then 
	    	printbuffer(1, dmm.defbuffer1.n, dmm.defbuffer1)
		end
	end

	function hardwareTrigger()
	    digio.writebit({hardware_output_trigger_pin}, 1)
	    delay({hardware_output_trigger_time})
	    digio.writebit({hardware_output_trigger_pin}, 0)
	end

	function manualTrigger(pin, triggerTime)
		digio.writebit(pin, 1)
		delay(triggerTime)
		digio.writebit(pin, 0)
	end

	function softwareTrigger()
	    dmm.dmm.measure.read()
	end

	function markSquare(Xmin, Xmax, Ymin, Ymax, pd, pps, nrepeat)
	    smu.smua.source.rangev = math.max(math.abs(Xmin), math.abs(Xmax))
	    smu.smub.source.rangev = math.max(math.abs(Ymin), math.abs(Ymax))
		local perimeter = 2*(Xmax-Xmin)+2*(Ymax-Ymin)
		local step=perimeter/pd
		local Nx = math.floor((Xmax-Xmin)/step)
		local Ny = math.floor((Ymax-Ymin)/step)
	    smu.smua.source.levelv = Xmin
	    smu.smub.source.levelv = Ymin
		for k=0,nrepeat-1 do
			for i = 0, Nx-1 do
					local x = Xmin + i * step
					smu.smua.source.levelv = x
					delay(1/pps)
			end
			for i = 0, Ny-1 do
					local y = Ymin + i * step
					smu.smub.source.levelv = y
					delay(1/pps)
			end
			for i = 0, Nx-1 do
					local x = Xmax - i * step
					smu.smua.source.levelv = x
					delay(1/pps)
			end
			for i = 0, Ny-1 do
					local y = Ymax - i * step
					smu.smub.source.levelv = y
					delay(1/pps)
			end
		end
	    print(1)
	    smu.smua.source.levelv = {end_x}
	    smu.smub.source.levelv = {end_y}
	end

	function lineScan(Smin, Smax, NS, fixed, scanAxis, settleTime)
	    local dS = (Smax - Smin) / (NS - 1)
		if scanAxis==0 then
			smuS=smu.smua
			smuF=smu.smub
		elseif scanAxis== 1 then
			smuS=smu.smub
			smuF=smu.smua
		else
			smu.beeper.beep(1,2000)
			return 
		end
	    smuS.source.rangev = math.max(math.abs(Smin), math.abs(Smax))
	    smuF.source.rangev = math.abs(fixed)
		smuF.source.levelv=fixed
		smuS.source.levelv=Smin
		delay(settleTime)
	    dmm.defbuffer1.capacity = NS
		for i = 0, NS-1 do
	        local s = Smin + i * dS
	        smuS.source.levelv = s
	        delay(settleTime)
	        dmm.dmm.measure.read()
	    end
	    print(1)
	    printbuffer(1, dmm.defbuffer1.n, dmm.defbuffer1)
		smuS.source.levelv={end_s}
		smuF.source.levelv={end_f}

	end
	function moveAxis(Smin, Smax, value, axis, settleTime)
		if axis==0 then
			smuS=smu.smua
		elseif axis==1 then
			smuS=smu.smub
		else
			smu.beeper.beep(1,2000)
			return
		end
		smuS.source.rangev = math.max(math.abs(Smin), math.abs(Smax))
		smuS.source.levelv = value
		delay(settleTime)
		print(1)
	end

	function readPoint()
		dmm.dmm.measure.read()
    	print(dmm.defbuffer1[dmm.defbuffer1.n])
	end
	
	function goTo(x, y)
		smua.source.rangev=x
		smub.source.rangev=y
		smua.source.levelv=x
		smub.source.levelv=y
	end
endscript
	"""

			for line in tsp_script.splitlines():
				self.k.write(line)
			self.k.write("makeFunc()")
			self.flashed=True
			for t in triggers_to_initialize:
				self.k.write(f"digio.writebit({t}, 0)")

		except Exception as e:
			print(e)
			print("Failed to flash the default software")
			return -1

		return 0
	def initiate(self):
		self.k.write('initiateInstrument()')

	def close(self):
		self.k.close()

	def fill_buffer_manual_trigger(self, trigger_pin, sampling, nAvg=1,  verbose=False, resetFct=None):

		out=np.zeros(sampling)
		trigger_pin=int(trigger_pin)
		for n in range(nAvg):
			if resetFct is not None:
				# print("Reset fct called")
				resetFct()

			self.verbose_write(f'fillBufferManualTrigger({trigger_pin}, {sampling})', verbose)
			sbuff=self.k.read()
			buff=np.fromstring(sbuff,sep=',')
			out+=buff
		return out/nAvg
	
	def go_to(self, x, y):
		x=self.enforce_limits_0D(x)
		y=self.enforce_limits_0D(y)
		self.k.write(f'goTo({x}, {y})')


	def raster_scan(self,coordinates, sampling, trigger_type, coordinate_basis="centered", settle_time=1e-4, return_line_settle_time=1e-3, verbose=False):
		if coordinate_basis not in ["centered", "framed"]:
			print('coordinate_basis must be either "centered" (xcenter, width, ycenter, height) or "framed" (xmin, xmax, ymin, ymax)')
			print("Exiting for safety")
			return -1
		if coordinate_basis=="centered":
			coordinates=centered_to_framed(*coordinates)
		
		coordinates=self.enforce_limits(*coordinates)
		if type(sampling)==int:
			sampling=[sampling, sampling]
		elif not (type(sampling)==list and len(sampling)==2):
			print(f"Cannot interpret {sampling} as a uniform or full sampling, please provide an int or a 2-elements list")
			return -1
		if sampling[0]*sampling[1]>16000 and bool((trigger_type>>1) & 1):
			print("Rastering more than 16000 points, consider using raster_scan_buffer_synced")
		if verbose:
			print(f"Scanning the following area (framed basis): {coordinates}")
			verbose_trigger_type(trigger_type)
		self.verbose_write(f"rasterScan({coordinates[0]}, {coordinates[1]}, {sampling[0]}, {coordinates[2]}, {coordinates[3]}, {sampling[1]}, {settle_time}, {return_line_settle_time}, {trigger_type})", verbose)
		self.k.read()
		waiting_time=max(round(sampling[0]*sampling[1]*(max(self.nplc/50, 1e-3)+settle_time),2),0)*0.85
		divider=waiting_time/50
		if verbose:
			print("Scan started")
			print(f"Waiting {waiting_time} s")
			for i in range(50):
				print('.', end='')
			print('')
		for i in range(50):
			time.sleep(divider)
			if verbose:
				print('.', end='')
		self.k.read()
		if verbose:
			print('')
			print('Scan done')
			if bool(trigger_type & 1):
				print("Querying")
		if bool(trigger_type & 1):
			sbuff=self.k.read()
			buff=np.fromstring(sbuff,sep=',').reshape((sampling[1],sampling[0]))
			return np.linspace(coordinates[0], coordinates[1], sampling[0]),np.linspace(coordinates[2], coordinates[3], sampling[1]), buff
		else:
			return np.linspace(coordinates[0], coordinates[1], sampling[0]),np.linspace(coordinates[2], coordinates[3], sampling[1]), 0


	def raster_scan_buffer_synced(self,coordinates, sampling, trigger_type, sr830, coordinate_basis="centered", buffer_size=16000, settle_time=1e-4, return_line_settle_time=1e-3, verbose=False):
		if not bool((trigger_type>>1) & 1):
			print('Buffer synced is only available for hardware trigger, rolling back to raster_scan')
			return self.raster_scan(coordinates, sampling, trigger_type,coordinate_basis, settle_time, return_line_settle_time, verbose)

		if coordinate_basis not in ["centered", "framed"]:
			print('coordinate_basis must be either "centered" (xcenter, width, ycenter, height) or "framed" (xmin, xmax, ymin, ymax)')
			print("Exiting for safety")
			return -1
		if coordinate_basis=="centered":
			coordinates=centered_to_framed(*coordinates)
		coordinates=self.enforce_limits(*coordinates)
		if type(sampling)==int:
			sampling=[sampling, sampling]
		elif not (type(sampling)==list and len(sampling)==2):
			print(f"Cannot interpret {sampling} as a uniform or full sampling, please provide an int or a 2-elements list")
			return -1
		if verbose:
			print(f"Scanning the following area (framed basis): {coordinates}")
			verbose_trigger_type(trigger_type)
		total_size=sampling[0]*sampling[1]

		full_buffer_X=np.empty((total_size))
		full_buffer_Y=np.empty((total_size))

		sr830.trigger_reset()
		sr830.trigger_start()
		time.sleep(1)
		self.verbose_write(f"rasterScanBufferSynced({coordinates[0]}, {coordinates[1]}, {sampling[0]}, {coordinates[2]}, {coordinates[3]}, {sampling[1]}, {settle_time}, {return_line_settle_time}, {trigger_type}, {buffer_size})", verbose)
		self.k.read()

		current_position_in_buffer=0
		for sub_buffer_index in range(np.ceil(sampling[0]*sampling[1]/buffer_size)):
			next_buffer_size=min(total_size-current_position_in_buffer, buffer_size)
			sub_waiting_time=round(next_buffer_size*(max(self.nplc/50, 1.8e-3)+settle_time),2)

			divider=int(sub_waiting_time/5)
			if verbose:
				if sub_buffer_index==0:
					print("Scan started")
				else:
					print("Continuing scan")
				print(f"Waiting {sub_waiting_time} s")
				for i in range(int(sub_waiting_time//divider)):
					print('.', end='')
				print('')
			for i in range(int(sub_waiting_time//divider)):
				time.sleep(divider)
				if verbose:
					print('.', end='')
			self.k.read()

			if verbose:
				print('')
			X=sr830.get_buffer_array_binary(1)
			Y=sr830.get_buffer_array_binary(2)

			full_buffer_X[current_position_in_buffer:current_position_in_buffer+X.size]=X
			full_buffer_Y[current_position_in_buffer:current_position_in_buffer+Y.size]=Y
			current_position_in_buffer+=X.size
			if verbose:
				print(f"Buffer {sub_buffer_index} saved ({current_position_in_buffer}/{sampling[0]*sampling[1]})")	

			sr830.trigger_reset()
			sr830.trigger_start()
			time.sleep(1)
			self.k.write('be=1')

		
		if verbose:
			print('')
			print('Scan done')
			if bool(trigger_type & 1):
				print("Querying")
		if bool(trigger_type & 1):
			sbuff=self.k.read()
			buff=np.fromstring(sbuff,sep=',').reshape((sampling[1],sampling[0]))
			return np.linspace(coordinates[0], coordinates[1], sampling[0]),np.linspace(coordinates[2], coordinates[3], sampling[1]), buff
		else:
			return np.linspace(coordinates[0], coordinates[1], sampling[0]),np.linspace(coordinates[2], coordinates[3], sampling[1]), 0


	def raster_scan_partitioned(self, coordinates, sampling, trigger_type, lockins,
		coordinate_basis="centered", settle_time=1e-4,
		return_line_settle_time=1e-3, verbose=False):
		# Normalize lockins input
		if not isinstance(lockins, (list, tuple)):
			lockins = [lockins]

		# 1. Coordinate Setup
		if coordinate_basis == "centered":
			coordinates = centered_to_framed(*coordinates)

		coordinates = self.enforce_limits(*coordinates)
		xmin, xmax, ymin, ymax = coordinates

		nx, ny = (sampling, sampling) if isinstance(sampling, int) else sampling

		# 2. Partition Calculation
		max_points = 15700
		lines_per_block = max_points // nx
		if lines_per_block == 0:
			raise ValueError(f"Nx ({nx}) exceeds buffer limit.")

		num_blocks = int(np.ceil(ny / lines_per_block))
		y_space = np.linspace(ymin, ymax, ny)

		# Preallocate
		num_li = len(lockins)
		LI_X = [np.zeros((ny, nx)) for _ in range(num_li)]
		LI_Y = [np.zeros((ny, nx)) for _ in range(num_li)]

		# 3. Execution Loop
		for i in range(num_blocks):
			start_idx = i * lines_per_block
			end_idx = min((i + 1) * lines_per_block, ny)
			current_ny = end_idx - start_idx

			sub_ymin = y_space[start_idx]
			sub_ymax = y_space[end_idx - 1]

			# Reset and start all lock-ins
			for li in lockins:
				li.trigger_reset()
				li.trigger_start()

			time.sleep(3)

			if verbose:
				print(f"Block {i+1}/{num_blocks}: Lines {start_idx} to {end_idx-1}")

			# Run scan
			self.raster_scan(
				[xmin, xmax, sub_ymin, sub_ymax],
				[nx, current_ny],
				trigger_type,
				coordinate_basis="framed",
				settle_time=settle_time,
				return_line_settle_time=return_line_settle_time,
				verbose=verbose
			)

			# Retrieve data
			try:
				for idx, li in enumerate(lockins):
					LI_X[idx][start_idx:end_idx, :] = li.get_buffer_array_binary(1).reshape((current_ny, nx))
					LI_Y[idx][start_idx:end_idx, :] = li.get_buffer_array_binary(2).reshape((current_ny, nx))

			except ValueError as e:
				print(f"Buffer Retrieval Error in block {i}: {e}")
				break

		x_axis = np.linspace(xmin, xmax, nx)

		# Final structure
		data = [[LI_X[i], LI_Y[i]] for i in range(num_li)]

		return x_axis, y_space, data

	def line_scan(self,sampling_range, sampling, fixed_coordinate, scan_axis, settle_time=1e-4):
		if not (type(sampling_range)==list and len(sampling_range)==2):
			print(f"Cannot interpret {sampling_range} as a sampling range, please provide a 2-elements list, exiting")
			return -1
		if not (type(fixed_coordinate)==float or type(fixed_coordinate)==int):
			print(f"Cannot interpret {fixed_coordinate} as a fixed coordinate, please provide a float/int, exiting")
			return -1
		sampling_range=self.enforce_limits_1D(*sampling_range)
		if scan_axis==0 or scan_axis=='x':
			scan_axis=0
		elif scan_axis==1 or scan_axis=='y':
			scan_axis=1
		else: 
			print(f"Cannot interpret {scan_axis} as a scan axis, please provide either 0/'x', or 1/'y'")


		if abs(fixed_coordinate)>self.voltage_limit:
			print(f"Reducing fixed_coordinate from {fixed_coordinate} to", end='')
			fixed_coordinate=np.sign(fixed_coordinate)*self.voltage_limit
			print(fixed_coordinate)
		self.k.write(f"lineScan({sampling_range[0]},{sampling_range[1]},{sampling},{fixed_coordinate},{scan_axis},{settle_time})")

		self.k.read()
		sbuff=self.k.read()
		buff=np.fromstring(sbuff,sep=',')
		return np.linspace(sampling_range[0], sampling_range[1], sampling), buff
	
	def mark_square(self,coordinates, coordinate_basis="centered", verbose=False, points_per_perimeter=100, points_per_second=1000, n_repeat=1):
		if coordinate_basis not in ["centered", "framed"]:
			print('coordinate_basis must be either "centered" (xcenter, width, ycenter, height) or "framed" (xmin, xmax, ymin, ymax)')
			print("Exiting for safety")
			return -1
		if coordinate_basis=="centered":
			coordinates=centered_to_framed(*coordinates)
		
		coordinates=self.enforce_limits(*coordinates)
		if verbose:
			print(f"Scanning the following area (framed basis): {coordinates}")
		self.verbose_write(f"markSquare({coordinates[0]}, {coordinates[1]}, {coordinates[2]}, {coordinates[3]}, {points_per_perimeter}, {points_per_second}, {n_repeat})", verbose)
		time.sleep(n_repeat*points_per_perimeter/points_per_second)
		self.k.read()
		return 0
	def move_axis(self, sampling_range, value, scan_axis, settle_time=1e-3):
		if not (type(sampling_range)==list and len(sampling_range)==2):
			print(f"Cannot interpret {sampling_range} as a sampling range, please provide a 2-elements list, exiting")
			return -1
		sampling_range = self.enforce_limits_1D(*sampling_range)

		if scan_axis==0 or scan_axis=='x':
			scan_axis=0
		elif scan_axis==1 or scan_axis=='y':
			scan_axis=1
		else:
			print(f"Cannot interpret {scan_axis} as a scan axis, please provide either 0/'x', or 1/'y'")
			return -1

		if abs(value) > self.voltage_limit:
			print(f"Reducing value from {value} to ", end='')
			value = np.sign(value)*self.voltage_limit
			print(value)

		self.k.write(f"moveAxis({sampling_range[0]},{sampling_range[1]},{value},{scan_axis},{settle_time})")
		self.k.read()


	def read_point(self):
		self.k.write("readPoint()")
		return float(self.k.read())

	def change_DMM_NPLC(self, NPLC):
		self.k.write(f"changeDMMNPLC({NPLC})")

	def change_DMM_range(self, range):
		self.k.write(f"changeDMMRange({range})")

	def manual_trigger(self, trigger_pin, trigger_time=1e-3):
		trigger_pin=int(trigger_pin)
		self.k.write(f"manualTrigger({trigger_pin}, {trigger_time})")


# SAVE OLD CODE
	"""

	function rasterScanTrigger(Xmin, Xmax, Nx, Ymin, Ymax, Ny, settleTime)
	    local dx = (Xmax - Xmin) / (Nx - 1)
	    local dy = (Ymax - Ymin) / (Ny - 1)
	    local sampleCount = Nx * Ny
	    smu.smua.source.rangev = math.max(math.abs(Xmin), math.abs(Xmax))
	    smu.smub.source.rangev = math.max(math.abs(Ymin), math.abs(Ymax))
	    dmm.defbuffer1.capacity = sampleCount
	    print("Starting")

	    for j = 0, Ny-1 do
	        local y = Ymin + j * dy
	        smu.smub.source.levelv = y
	        for i = 0, Nx-1 do
	            local x = Xmin + i * dx
	            smu.smua.source.levelv = x
	        	--smu.beeper.beep(settleTime,440)
	        	delay(settleTime)
	        	digio.writebit({hardware_output_trigger_pin}, 1)
	        	delay(0.001)
	        	digio.writebit({hardware_output_trigger_pin}, 0)
	        end
	    end
		smu.smua.source.levelv = {end_x}
	    smu.smub.source.levelv = {end_y}
	    print("Done")
	end
	"""