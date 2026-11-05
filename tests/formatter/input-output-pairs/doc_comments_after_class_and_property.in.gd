class Inner:
	pass

## A constant documented after a class.
const A = 1

var prop:
	get:
		return 1

## A function documented after a property.
func f():
	pass

class Second:
	pass

# A plain comment before a variable.
var b = 2

class Third:
	pass

	## Kept inside the class.