use crate::{
    dynamics::solver::{
        solver_body::{SolverBody, SolverBodyInertia},
        xpbd::*,
    },
    prelude::*,
};
use bevy::prelude::*;

/// Constraint data required by the XPBD constraint solver for a fixed angle constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serialize", reflect(Serialize, Deserialize))]
#[reflect(Debug, PartialEq)]
pub struct FixedAngleConstraintShared {
    /// The target rotation difference between the two bodies.
    #[cfg(feature = "2d")]
    pub rotation_difference: Scalar,
    /// The target rotation difference between the two bodies.
    #[cfg(feature = "3d")]
    pub rotation_difference: Quaternion,
    /// The total Lagrange multiplier across the whole time step.
    pub total_lagrange: AngularVector,
}

impl XpbdConstraintSolverData for FixedAngleConstraintShared {
    fn clear_lagrange_multipliers(&mut self) {
        self.total_lagrange = AngularVector::ZERO;
    }

    fn total_rotation_lagrange(&self) -> AngularVector {
        self.total_lagrange
    }
}

impl FixedAngleConstraintShared {
    /// Prepares the constraint with the given rotations and local basis orientations.
    pub fn prepare(
        &mut self,
        rotation1: &Rotation,
        rotation2: &Rotation,
        local_basis1: Rot,
        local_basis2: Rot,
    ) {
        // Prepare the base rotation difference.
        #[cfg(feature = "2d")]
        {
            self.rotation_difference =
                (*rotation1 * local_basis1).angle_between(*rotation2 * local_basis2);
        }
        #[cfg(feature = "3d")]
        {
            self.rotation_difference =
                (rotation1.0 * local_basis1) * (rotation2.0 * local_basis2).inverse();
        }
    }

    /// Solves the constraint for the given bodies.
    pub fn solve(
        &mut self,
        bodies: [&mut SolverBody; 2],
        inertias: [&SolverBodyInertia; 2],
        compliance: Scalar,
        dt: Scalar,
    ) {
        let [body1, body2] = bodies;
        let [inertia1, inertia2] = inertias;

        let inv_inertia1 = inertia1.effective_inv_angular_inertia();
        let inv_inertia2 = inertia2.effective_inv_angular_inertia();

        #[cfg(feature = "2d")]
        let difference =
            self.rotation_difference + body1.delta_rotation.angle_between(body2.delta_rotation);
        #[cfg(feature = "3d")]
        // TODO: The XPBD paper doesn't have this minus sign, but it seems to be needed for stability.
        //       The angular correction code might have a wrong sign elsewhere.
        let difference = -2.0
            * (self.rotation_difference
                * body1.delta_rotation.0
                * body2.delta_rotation.0.inverse())
            .xyz();

        #[cfg(feature = "2d")]
        {
            // Align orientation
            self.total_lagrange += self.align_orientation(
                body1,
                body2,
                inv_inertia1,
                inv_inertia2,
                difference,
                0.0,
                compliance,
                dt,
            );
        }

        #[cfg(feature = "3d")]
        {
            self.total_lagrange += self.align_orientation_block(
                body1,
                body2,
                inv_inertia1,
                inv_inertia2,
                difference,
                compliance,
                dt,
            );
        }
    }

    /// Aligns the orientation of the bodies by solving all three rotational degrees of freedom
    /// **together**, as one 3x3 system.
    ///
    /// [`AngularConstraint::align_orientation`] treats the rotation error as a single axis `n` and
    /// solves one scalar along it: `w = n·I⁻¹·n`, `Δλ = −θ/(w + α̃)`, and each body turns by
    /// `I⁻¹·n·Δλ`. That is exact along `n` and nowhere else, because `I⁻¹·n` is not parallel to `n`
    /// unless `n` happens to be a principal axis. For a body whose inertia is very unequal between
    /// axes — a slender rod is thousands of times easier to spin about its length than end over
    /// end — a small twist component in `n` dominates `w`: the twist is over-corrected by roughly
    /// `1/n_twist²` of itself and the bending is under-corrected by the same factor. The overshoot
    /// is the next substep's error, mostly twist, and the joint whips the body about its own axis
    /// instead of converging.
    ///
    /// The block solve has no preferred axis to overshoot along:
    ///
    /// `Δλ = −(I₁⁻¹ + I₂⁻¹ + α̃·I)⁻¹ · θ`, each body turned by `±I⁻¹·Δλ`
    ///
    /// which drives the linearized error to exactly what the compliance allows on all three axes
    /// at once. When the effective mass matrix is singular — both bodies static, or rotation locked
    /// on an axis with a rigid joint — it falls back to the single-axis solve.
    ///
    /// Returns the Lagrange multiplier update, in the same convention as `align_orientation`.
    #[cfg(feature = "3d")]
    #[allow(clippy::too_many_arguments)]
    fn align_orientation_block(
        &self,
        body1: &mut SolverBody,
        body2: &mut SolverBody,
        inv_inertia1: SymmetricTensor,
        inv_inertia2: SymmetricTensor,
        rotation_difference: Vector,
        compliance: Scalar,
        dt: Scalar,
    ) -> AngularVector {
        if rotation_difference.length_squared() <= Scalar::EPSILON * Scalar::EPSILON {
            return AngularVector::ZERO;
        }

        let tilde_compliance = compliance / dt.powi(2);
        let effective = inv_inertia1
            + inv_inertia2
            + SymmetricTensor::from_diagonal(Vector::splat(tilde_compliance));

        // A matrix that cannot be inverted is one with a direction nothing can turn along.
        let scale = effective.diagonal().max_element();
        let determinant = effective.determinant();
        if !(scale > 0.0) || !determinant.is_finite() || determinant <= scale.powi(3) * 1.0e-9 {
            return self.align_orientation(
                body1,
                body2,
                inv_inertia1,
                inv_inertia2,
                rotation_difference,
                0.0,
                compliance,
                dt,
            );
        }

        // The impulse the scalar path would apply as `−Δλ·n`, now a full vector.
        let impulse = effective.inverse() * rotation_difference;
        self.apply_angular_impulse(body1, body2, inv_inertia1, inv_inertia2, impulse);

        // `align_orientation` returns `Δλ·n`, which is `−impulse`.
        -impulse
    }
}

impl AngularConstraint for FixedAngleConstraintShared {}
