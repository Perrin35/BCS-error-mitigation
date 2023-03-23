OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4794[3];
rz(3.0607858) q[0];
sx q[0];
rz(-1.6080987) q[0];
sx q[0];
rz(-0.62961271) q[0];
rz(1.1687006) q[1];
sx q[1];
rz(5.909579) q[1];
sx q[1];
rz(12.49514) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
rz(1.1761551) q[2];
sx q[2];
rz(-2.9784456) q[2];
sx q[2];
rz(2.0102248) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.62961271) q[0];
sx q[0];
rz(-1.533494) q[0];
sx q[0];
rz(0.080806832) q[0];
rz(1.1313678) q[1];
sx q[1];
rz(-2.9784456) q[1];
sx q[1];
rz(-3.0703621) q[2];
sx q[2];
rz(-0.37360631) q[2];
sx q[2];
rz(-1.1687006) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4794[0];
measure q[1] -> c4794[1];
measure q[2] -> c4794[2];
