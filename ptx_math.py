"""Our own scalar PTX math expansions; no external compiler or math library.

These are accuracy-oriented baselines, not correctly-rounded libm replacements.
The mixin uses PTXEmitter's registers/literals/labels. See docs/compiler/11-operation-expansion.md.
"""

import math


class ScalarMath:
    def activation(self, operation, value, dtype):
        """Native half tanh; accurate exp-based f32/f64 activations."""
        if operation == "tanh" and dtype == "f16":
            result = self.reg("f")
            self.emit(f"tanh.approx.f32 {result}, {value};")
            return result
        kind = "f64" if dtype == "f64" else "f32"
        bank = "d" if dtype == "f64" else "f"
        lit = lambda x: self.literal(x, kind)
        absolute, negative = self.reg(bank), self.reg(bank)
        self.emit(f"abs.{kind} {absolute}, {value};")
        self.emit(f"mul.rn.{kind} {negative}, {absolute}, {lit(-2. if operation == 'tanh' else -1.)};")
        t = self.exponential64(negative) if dtype == "f64" else self.native_math32("exp", negative)
        denominator = self.reg(bank)
        self.emit(f"add.rn.{kind} {denominator}, {lit(1.)}, {t};")
        result, opposite = self.reg(bank), self.reg(bank)
        choose = self.reg("p")
        if operation == "sigmoid":
            # Only exponentiate nonpositive arguments. Avoid overflow for x<0
            # and avoid cancellation from computing 1-sigmoid(abs(x)).
            self.emit(f"div.rn.{kind} {result}, {lit(1.)}, {denominator};")
            self.emit(f"div.rn.{kind} {opposite}, {t}, {denominator};")
            self.emit(f"setp.ge.{kind} {choose}, {value}, {lit(0.)};")
            self.emit(f"selp.{kind} {result}, {result}, {opposite}, {choose};")
        else:
            numerator = self.reg(bank)
            self.emit(f"sub.rn.{kind} {numerator}, {lit(1.)}, {t};")
            self.emit(f"div.rn.{kind} {result}, {numerator}, {denominator};")
            # The exp ratio cancels near zero. Use the odd tanh series there,
            # keeping signed zero and subnormal inputs through the final x*P.
            done = self.label("TANH_SERIES_DONE")
            self.emit(f"setp.lt.{kind} {choose}, {absolute}, {lit(.125)};")
            self.emit(f"@!{choose} bra {done};")
            square, polynomial = self.reg(bank), self.reg(bank)
            self.emit(f"mul.rn.{kind} {square}, {absolute}, {absolute};")
            coefficients = (-929569/638512875, 21844/6081075, -1382/155925,
                            62/2835, -17/315, 2/15, -1/3, 1.)
            if dtype == "f32":
                coefficients = coefficients[-4:]  # Degree seven suffices here.
            self.emit(f"mov.{kind} {polynomial}, {lit(coefficients[0])};")
            for coefficient in coefficients[1:]:
                self.emit(f"fma.rn.{kind} {polynomial}, {polynomial}, {square}, {lit(coefficient)};")
            self.emit(f"mul.rn.{kind} {result}, {absolute}, {polynomial};")
            self.lines.append(f"{done}:")
            self.emit(f"neg.{kind} {opposite}, {result};")
            width = 64 if dtype == "f64" else 32
            bits = self.reg("l" if width == 64 else "r")
            self.emit(f"mov.b{width} {bits}, {value};")
            self.emit(f"setp.lt.s{width} {choose}, {bits}, 0;")
            self.emit(f"selp.{kind} {result}, {opposite}, {result}, {choose};")
        return result

    def native_math32(self, operation, value):
        """Use PTX's f32 exp2/log2 instructions, without flush-to-zero."""
        result = self.reg("f")
        if operation in ("exp", "exp_fast", "exp2"):
            if operation == "exp2":
                self.emit(f"ex2.approx.f32 {result}, {value};")
                return result
            # Multiplying a large x by rounded log2(e) alone loses too much
            # accuracy. Reduce x=k*ln(2)+r, then ex2(r*log2(e))*ex2(k).
            scaled, exponent, negative, residual, fraction, scale = (self.reg("f") for _ in range(6))
            self.emit(f"mul.rn.f32 {scaled}, {value}, {self.literal(math.log2(math.e), 'f32')};")
            self.emit(f"cvt.rni.f32.f32 {exponent}, {scaled};")
            # Keep the scale normal and finite; the final multiply handles
            # subnormal outputs and overflow without premature infinity.
            self.emit(f"max.f32 {exponent}, {exponent}, {self.literal(-126., 'f32')};")
            self.emit(f"min.f32 {exponent}, {exponent}, {self.literal(127., 'f32')};")
            self.emit(f"neg.f32 {negative}, {exponent};")
            self.emit(f"fma.rn.f32 {residual}, {negative}, {self.literal(0.6931471824645996, 'f32')}, {value};")
            self.emit(f"fma.rn.f32 {residual}, {negative}, {self.literal(-1.904654299957768e-9, 'f32')}, {residual};")
            self.emit(f"mul.rn.f32 {fraction}, {residual}, {self.literal(math.log2(math.e), 'f32')};")
            self.emit(f"ex2.approx.f32 {result}, {fraction};")
            self.emit(f"ex2.approx.f32 {scale}, {exponent};")
            self.emit(f"mul.rn.f32 {result}, {result}, {scale};")
            # At the half-min-subnormal boundary, ex2's allowed error can
            # turn a just-positive result into an exact tie rounded to zero.
            # This rounded cutoff is the first f32 input whose exp is nonzero.
            nonzero = self.reg("p")
            self.emit(f"setp.ge.f32 {nonzero}, {value}, {self.literal(-150.*math.log(2.), 'f32')};")
            self.emit(f"@{nonzero} max.f32 {result}, {result}, 0f00000001;")
        else:
            self.emit(f"lg2.approx.f32 {result}, {value};")
            done = self.label("LOG_NATIVE_DONE")
            near, upper = self.reg("p"), self.reg("p")
            self.emit(f"setp.ge.f32 {near}, {value}, {self.literal(.75, 'f32')};")
            self.emit(f"setp.le.f32 {upper}, {value}, {self.literal(1.25, 'f32')};")
            self.emit(f"and.pred {near}, {near}, {upper};")
            self.emit(f"@!{near} bra {done};")
            # lg2's absolute-error bound is inadequate when ln(x) approaches
            # zero. A short odd series in z=(x-1)/(x+1) is accurate here.
            numerator, denominator, z, square, polynomial = (self.reg("f") for _ in range(5))
            self.emit(f"sub.rn.f32 {numerator}, {value}, {self.literal(1., 'f32')};")
            self.emit(f"add.rn.f32 {denominator}, {value}, {self.literal(1., 'f32')};")
            self.emit(f"div.rn.f32 {z}, {numerator}, {denominator};")
            self.emit(f"mul.rn.f32 {square}, {z}, {z};")
            self.emit(f"mov.f32 {polynomial}, {self.literal(1./9., 'f32')};")
            for odd in (7, 5, 3, 1):
                self.emit(f"fma.rn.f32 {polynomial}, {polynomial}, {square}, {self.literal(1./odd, 'f32')};")
            self.emit(f"mul.rn.f32 {z}, {z}, {self.literal(2., 'f32')};")
            self.emit(f"mul.rn.f32 {polynomial}, {polynomial}, {z};")
            # Keep the common result in log2 units before the base conversion.
            self.emit(f"mul.rn.f32 {result}, {polynomial}, {self.literal(math.log2(math.e), 'f32')};")
            self.lines.append(f"{done}:")
            if operation != "log2":
                scale = math.log(2.) if operation == "log" else math.log10(2.)
                scaled = self.reg("f")
                self.emit(f"mul.rn.f32 {scaled}, {result}, {self.literal(scale, 'f32')};")
                result = scaled
        return result

    def integer_power(self, base, exponent, dtype):
        """Exponentiation by squaring for valid nonnegative integer exponents.

        Multiplication keeps the declared width, matching integer wraparound.
        Negative integer exponents are outside this operation's input domain.
        """
        kind = self.ptx_type(dtype)
        width = self.element_bytes(dtype)*8
        result, factor, remaining, low = (self.data_reg(dtype) for _ in range(4))
        finished, odd = self.reg("p"), self.reg("p")
        head, done = self.label("IPOW"), self.label("IPOW_DONE")
        self.emit(f"mov.{kind} {result}, 1;")
        self.emit(f"mov.{kind} {factor}, {base};")
        self.emit(f"mov.{kind} {remaining}, {exponent};")
        self.lines.append(f"{head}:")
        self.emit(f"setp.eq.{kind} {finished}, {remaining}, 0;")
        self.emit(f"@{finished} bra {done};")
        self.emit(f"and.b{width} {low}, {remaining}, 1;")
        self.emit(f"setp.ne.{kind} {odd}, {low}, 0;")
        self.emit(f"@{odd} mul.lo.{kind} {result}, {result}, {factor};")
        self.emit(f"mul.lo.{kind} {factor}, {factor}, {factor};")
        self.emit(f"shr.u{width} {remaining}, {remaining}, 1;")
        self.emit(f"bra {head};")
        self.lines.append(f"{done}:")
        return result

    def math_binary64(self, opcode, a, b):
        result = self.reg("d")
        self.emit(f"{opcode}.rn.f64 {result}, {a}, {b};")
        return result

    def math_fma64(self, a, b, c):
        result = self.reg("d")
        self.emit(f"fma.rn.f64 {result}, {a}, {b}, {c};")
        return result

    def exponential64(self, value):
        """exp(x): x=k*ln(2)+r, Taylor on |r|<=ln(2)/2, scaled by 2**k."""
        lit = lambda x: self.literal(x, "f64")
        result, predicate = self.reg("d"), self.reg("p")
        done = self.label("EXP_DONE")
        # The selected cutoffs are outside the transition intervals. They
        # also keep conversion to the integer range-reduction exponent safe.
        for comparison, limit, answer in (("gt", 710., math.inf), ("lt", -746., 0.)):
            self.emit(f"setp.{comparison}.f64 {predicate}, {value}, {lit(limit)};")
            self.emit(f"@{predicate} mov.f64 {result}, {lit(answer)};")
            self.emit(f"@{predicate} bra {done};")
        self.emit(f"setp.nan.f64 {predicate}, {value}, {value};")
        self.emit(f"@{predicate} mov.f64 {result}, {value};")
        self.emit(f"@{predicate} bra {done};")
        scaled = self.math_binary64("mul", value, lit(math.log2(math.e)))
        k, kd = self.reg("r"), self.reg("d")
        self.emit(f"cvt.rni.s32.f64 {k}, {scaled};")
        self.emit(f"cvt.rn.f64.s32 {kd}, {k};")
        negative = self.reg("d")
        self.emit(f"neg.f64 {negative}, {kd};")
        # Split ln(2) avoids losing the residual when x is large.
        residual = self.math_fma64(negative, lit(float.fromhex("0x1.62e42fee00000p-1")), value)
        residual = self.math_fma64(negative, lit(1.9082149292705877e-10), residual)
        polynomial = lit(1. / math.factorial(18))
        for degree in range(17, -1, -1):
            polynomial = self.math_fma64(polynomial, residual, lit(1. / math.factorial(degree)))
        # Split the power of two into two normal factors. Subnormals are
        # rounded on the final multiply rather than in an intermediate scale.
        clamped, remainder = self.reg("r"), self.reg("r")
        self.emit(f"max.s32 {clamped}, {k}, -1022;")
        self.emit(f"min.s32 {clamped}, {clamped}, 1023;")
        self.emit(f"sub.s32 {remainder}, {k}, {clamped};")
        factors = []
        for exponent in (clamped, remainder):
            biased, bits, factor = self.reg("r"), self.reg("l"), self.reg("d")
            self.emit(f"add.s32 {biased}, {exponent}, 1023;")
            self.emit(f"cvt.u64.u32 {bits}, {biased};")
            self.emit(f"shl.b64 {bits}, {bits}, 52;")
            self.emit(f"mov.b64 {factor}, {bits};")
            factors.append(factor)
        intermediate = self.math_binary64("mul", polynomial, factors[1])
        self.emit(f"mul.rn.f64 {result}, {intermediate}, {factors[0]};")
        self.lines.append(f"{done}:")
        return result

    def logarithm64(self, value):
        """ln(x): binary decomposition and 2*(z+z**3/3+...), |z|<0.172."""
        lit = lambda x: self.literal(x, "f64")
        tiny, scaled, multiplier = self.reg("p"), self.reg("d"), self.reg("d")
        self.emit(f"setp.lt.f64 {tiny}, {value}, {lit(float.fromhex('0x1p-1022'))};")
        self.emit(f"selp.f64 {multiplier}, {lit(2.**54)}, {lit(1.)}, {tiny};")
        self.emit(f"mul.rn.f64 {scaled}, {value}, {multiplier};")
        bits, exponent_bits = self.reg("l"), self.reg("l")
        self.emit(f"mov.b64 {bits}, {scaled};")
        self.emit(f"shr.u64 {exponent_bits}, {bits}, 52;")
        self.emit(f"and.b64 {exponent_bits}, {exponent_bits}, 2047;")
        exponent, correction = self.reg("r"), self.reg("r")
        self.emit(f"cvt.s32.u64 {exponent}, {exponent_bits};")
        self.emit(f"sub.s32 {exponent}, {exponent}, 1023;")
        self.emit(f"selp.s32 {correction}, 54, 0, {tiny};")
        self.emit(f"sub.s32 {exponent}, {exponent}, {correction};")
        mantissa = self.reg("d")
        self.emit(f"and.b64 {bits}, {bits}, 0x000fffffffffffff;")
        self.emit(f"or.b64 {bits}, {bits}, 0x3ff0000000000000;")
        self.emit(f"mov.b64 {mantissa}, {bits};")
        larger, halved = self.reg("p"), self.reg("d")
        self.emit(f"setp.gt.f64 {larger}, {mantissa}, {lit(math.sqrt(2.))};")
        self.emit(f"mul.rn.f64 {halved}, {mantissa}, {lit(.5)};")
        self.emit(f"selp.f64 {mantissa}, {halved}, {mantissa}, {larger};")
        self.emit(f"selp.s32 {correction}, 1, 0, {larger};")
        self.emit(f"add.s32 {exponent}, {exponent}, {correction};")
        numerator = self.math_binary64("sub", mantissa, lit(1.))
        denominator = self.math_binary64("add", mantissa, lit(1.))
        z = self.math_binary64("div", numerator, denominator)
        square = self.math_binary64("mul", z, z)
        polynomial = lit(1. / 27.)
        for odd in range(25, 0, -2):
            polynomial = self.math_fma64(polynomial, square, lit(1. / odd))
        series = self.math_binary64("mul", self.math_binary64("mul", lit(2.), z), polynomial)
        ed = self.reg("d")
        self.emit(f"cvt.rn.f64.s32 {ed}, {exponent};")
        result = self.math_fma64(ed, lit(math.log(2.)), series)
        pred = self.reg("p")
        for comparison, operand, answer in (("eq", 0., -math.inf), ("lt", 0., math.nan),
                                             ("eq", math.inf, math.inf)):
            self.emit(f"setp.{comparison}.f64 {pred}, {value}, {lit(operand)};")
            self.emit(f"@{pred} mov.f64 {result}, {lit(answer)};")
        self.emit(f"setp.nan.f64 {pred}, {value}, {value};")
        self.emit(f"@{pred} mov.f64 {result}, {value};")
        return result

    def floating_math(self, operation, operands, dtype):
        """Use native f32 exp/log for f16/f32; preserve double math for f64."""
        if dtype in ("f16", "f32") and operation in ("exp", "exp_fast", "exp2", "log", "log2", "log10"):
            # f16 operands already live in widened f32 registers. arithmetic()
            # retains its final half rounding at this IR operation boundary.
            return self.native_math32(operation, operands[0])
        args = [self.convert(value, dtype, "f64") for value in operands]
        if operation in ("exp", "exp_fast", "exp2"):
            value = args[0]
            if operation == "exp2":
                value = self.math_binary64("mul", value, self.literal(math.log(2.), "f64"))
            result = self.exponential64(value)
        elif operation in ("log", "log2", "log10"):
            result = self.logarithm64(args[0])
            if operation != "log":
                divisor = math.log(2.) if operation == "log2" else math.log(10.)
                result = self.math_binary64("div", result, self.literal(divisor, "f64"))
        elif operation in ("power", "pow", "float_power"):
            result = self.power64(*args)
        else:
            raise NotImplementedError(operation)
        # arithmetic() applies the usual f16 rounding after this conversion.
        return result if dtype == "f64" else self.convert(result, "f64", "f32")

    def power64(self, base, exponent):
        """Real-valued power, with domain/sign and zero/NaN/infinity fixups."""
        lit = lambda x: self.literal(x, "f64")
        absolute, truncated, half, floor_half, parity = (self.reg("d") for _ in range(5))
        self.emit(f"abs.f64 {absolute}, {base};")
        self.emit(f"cvt.rzi.f64.f64 {truncated}, {exponent};")
        self.emit(f"mul.rn.f64 {half}, {exponent}, {lit(.5)};")
        self.emit(f"cvt.rmi.f64.f64 {floor_half}, {half};")
        self.emit(f"fma.rn.f64 {parity}, {floor_half}, {lit(-2.)}, {exponent};")
        integer, odd, negative, invalid, predicate = (self.reg("p") for _ in range(5))
        self.emit(f"setp.eq.f64 {integer}, {truncated}, {exponent};")
        self.emit(f"setp.eq.f64 {odd}, {parity}, {lit(1.)};")
        # Read the sign bit so odd powers of -0 also get the right sign.
        bits = self.reg("l")
        self.emit(f"mov.b64 {bits}, {base};")
        self.emit(f"setp.lt.s64 {negative}, {bits}, 0;")
        logarithm = self.logarithm64(absolute)
        result = self.exponential64(self.math_binary64("mul", exponent, logarithm))
        self.emit(f"and.pred {predicate}, {negative}, {odd};")
        self.emit(f"@{predicate} neg.f64 {result}, {result};")
        self.emit(f"setp.lt.f64 {negative}, {base}, {lit(0.)};")
        self.emit(f"not.pred {invalid}, {integer};")
        self.emit(f"and.pred {invalid}, {invalid}, {negative};")
        # Negative infinity permits noninteger exponents; finite negative bases don't.
        self.emit(f"setp.ne.f64 {predicate}, {absolute}, {lit(math.inf)};")
        self.emit(f"and.pred {invalid}, {invalid}, {predicate};")
        self.emit(f"@{invalid} mov.f64 {result}, {lit(math.nan)};")
        # |base|==1 with infinite exponent is 1, even though inf*log(1) is NaN.
        self.emit(f"abs.f64 {half}, {exponent};")
        self.emit(f"setp.eq.f64 {predicate}, {half}, {lit(math.inf)};")
        self.emit(f"setp.eq.f64 {negative}, {absolute}, {lit(1.)};")
        self.emit(f"and.pred {predicate}, {predicate}, {negative};")
        self.emit(f"@{predicate} mov.f64 {result}, {lit(1.)};")
        # pow(NaN,0) and pow(1,NaN) are 1 under NumPy/PyTorch conventions.
        self.emit(f"setp.eq.f64 {predicate}, {exponent}, {lit(0.)};")
        self.emit(f"@{predicate} mov.f64 {result}, {lit(1.)};")
        self.emit(f"setp.eq.f64 {predicate}, {base}, {lit(1.)};")
        self.emit(f"@{predicate} mov.f64 {result}, {lit(1.)};")
        return result
