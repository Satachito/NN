//
//  main.swift
//  Section5
//
//  Created by Satoru Ogura on 2018/11/25.
//  Copyright © 2018 Satoru Ogura. All rights reserved.
//

import Foundation

struct
MulLayer< FP: Numeric > {
	var	x: FP = 0
	var	y: FP = 0
	mutating func
	Forward( _ x: FP, _ y: FP ) -> FP {
		self.x = x
		self.y = y
		return x * y
	}
	func
	Backward( _ d: FP ) -> ( FP, FP ) {
		return ( d * y, d * x )
	}
}

struct
AddLayer< FP: Numeric > {
	func
	Forward( _ x: FP, _ y: FP ) -> FP {
		return x + y
	}
	func
	Backward( _ d: FP ) -> ( FP, FP ) {
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

