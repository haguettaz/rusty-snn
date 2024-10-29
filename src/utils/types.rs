#[derive(Debug, Clone, Copy)]
pub struct Real(f64);

impl Real {
    pub fn new(value: f64) -> Real {
        if !value.is_finite() {
            panic!("Real value must be finite");
        }

        return Real(value);
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

impl std::ops::Add<Real> for Real {
    type Output = Real;

    fn add(self, other: Real) -> Real {
        Real(self.0 + other.0)
    }
}

impl std::ops::Sub<Real> for Real {
    type Output = Real;

    fn sub(self, other: Real) -> Real {
        Real(self.0 - other.0)
    }
}

impl PartialEq for Real {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Real {}

impl PartialOrd for Real {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PosReal(f64);

impl PosReal {
    pub fn new(value: f64) -> PosReal {
        if value <= 0.0 {
            panic!("Positive real value must be positive");
        }

        return PosReal(value);
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}

impl std::ops::Add<PosReal> for PosReal {
    type Output = PosReal;

    fn add(self, other: PosReal) -> PosReal {
        PosReal(self.0 + other.0)
    }
}

impl PartialEq for PosReal {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for PosReal {}

impl PartialOrd for PosReal {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
