import libmushu

# look for amplifiers connected to the system, and return a list of the
# respective classes
available_amps = libmushu.get_available_amps()

# select the first available amp and decorate it with tcp-marker- and
# save-to-file-functionality
ampname = available_amps[0]
amp = libmushu.get_amp(ampname)

# configure the amplifier
amp.configure(cfg)

# start it and collect data until finished
amp.start()
while 1:
    data, trigger = amp.get_data()

# stop the amplifier
amp.stop()
