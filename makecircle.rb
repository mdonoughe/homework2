#!/usr/bin/env ruby
3.times do |h|
  50.times do |i|
    angle = i * 2 * Math::PI / 50.to_f
    radius = h * 3 + 1
    puts "#{Math.cos(angle) * radius}\t#{Math.sin(angle) * radius}\t#{h % 2}"
  end
end
