import Foundation

struct
MulLayer< N: Numeric > {
	var	x: N = 0
	var	y: N = 0
	mutating func
	Forward( _ x: N, _ y: N ) -> N {
		self.x = x
		self.y = y
		return x * y
	}
	func
	Backward( _ d: N ) -> ( N, N ) {
		return ( d * y, d * x )
	}
}

struct
AddLayer< N: Numeric > {
	func
	Forward( _ x: N, _ y: N ) -> N {
		return x + y
	}
	func
	Backward( _ d: N ) -> ( N, N ) {
		return ( d, d )
	}
}

var	AppleMulLayer = MulLayer< Double >()
var	OrangeMulLayer = MulLayer< Double >()
var	AppleOrangeAddLayer = AddLayer< Double >()
var	VATMulLayer = MulLayer< Double >()

let	uApple = 100.0
let	nApple = 2.0
let	uOrange = 150.0
let	nOrange = 3.0
let	vat = 1.1

let	appleSum = AppleMulLayer.Forward( uApple, nApple )
let	orangeSum = OrangeMulLayer.Forward( uOrange, nOrange )
let	sum = AppleOrangeAddLayer.Forward( appleSum, orangeSum )
let	toPay = VATMulLayer.Forward( sum, vat )

print( appleSum, orangeSum, sum, toPay )

let	dPrice = 1.0
let	( dSum, dVAT ) = VATMulLayer.Backward( dPrice )
let	( dAppleSum, dOrangeSum ) = AppleOrangeAddLayer.Backward( dSum )
let ( dUApple, dNApple ) = AppleMulLayer.Backward( dAppleSum )
let ( dUOrange, dNOrange ) = OrangeMulLayer.Backward( dOrangeSum )

print( dSum, dVAT, dAppleSum, dOrangeSum, dUApple, dNApple, dUOrange, dNOrange )

