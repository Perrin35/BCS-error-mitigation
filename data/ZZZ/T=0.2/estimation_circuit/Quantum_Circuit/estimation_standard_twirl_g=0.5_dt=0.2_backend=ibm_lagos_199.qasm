OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4699[3];
rz(4.2498536) q[0];
sx q[0];
rz(4.9183283) q[0];
sx q[0];
rz(11.315622) q[0];
rz(0.91454084) q[1];
sx q[1];
rz(-0.17600888) q[1];
sx q[1];
rz(-0.25093719) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
rz(-1.2152018) q[2];
sx q[2];
rz(-0.95050496) q[2];
sx q[2];
rz(0.834922) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.890844) q[0];
sx q[0];
rz(-1.7767357) q[0];
sx q[0];
rz(-1.108261) q[0];
rz(2.3066707) q[1];
sx q[1];
rz(0.95050496) q[1];
sx q[1];
rz(2.8906555) q[2];
sx q[2];
rz(-2.9655838) q[2];
sx q[2];
rz(2.2270518) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4699[0];
measure q[1] -> c4699[1];
measure q[2] -> c4699[2];