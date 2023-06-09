OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4562[3];
rz(-0.55303108) q[0];
sx q[0];
rz(-0.49285046) q[0];
sx q[0];
rz(0.79883222) q[0];
rz(5.9981744) q[1];
sx q[1];
rz(3.7113037) q[1];
sx q[1];
rz(11.41834) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
rz(-0.84868969) q[2];
sx q[2];
rz(-1.797712) q[2];
sx q[2];
rz(-2.7925128) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.79883222) q[0];
sx q[0];
rz(-0.49285046) q[0];
sx q[0];
rz(0.55303108) q[0];
rz(0.34907985) q[1];
sx q[1];
rz(1.3438806) q[1];
sx q[1];
rz(-1.9935622) q[2];
sx q[2];
rz(-2.5718816) q[2];
sx q[2];
rz(0.28501093) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4562[0];
measure q[1] -> c4562[1];
measure q[2] -> c4562[2];
