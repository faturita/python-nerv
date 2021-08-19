 @tf.function
 def damage_city(severity, resources_allocated, time_damage_accumulates):
     # Not sure I interpreted correctly how damage is calculate, please check
     # option 1
     #damage = tf.math.maximum(0., severity - resources_allocated) * time_damage_accumulates
     # option 2
     #damage = tf.math.maximum(0., severity - resources_allocated) ** time_damage_accumulates
     # option 3 damage does not accumulate
     #damage = tf.math.maximum(0., severity - resources_allocated) ** 2
     # option 4 (revised based on experiment)
     α = 0.2
     ##damage = tf.math.maximum(0., (severity - resources_allocated) * (1 + α)**time_damage_accumulates + resources_allocated)
     damage = tf.math.maximum(0., (severity * (1 + α)-resources_allocated * α * time_damage_accumulates))
     return damage

severities = tf.constant([3,4,8], dtype=tf.float32)
resources = tf.constant([5,6,4], dtype=tf.float32)
city_number = tf.Variable(1.)
damage = tf.Variable(0.)
for k in range(len(severities)):
    sev = severities[k]
    time_damage_accumulates = len(severities) + 1 - city_number
    dam = damage_city(sev, resources[k], time_damage_accumulates)
    print('Time %d Damage: %10.5f' % (time_damage_accumulates, dam))
    damage = damage + dam 
    city_number = city_number + 1

