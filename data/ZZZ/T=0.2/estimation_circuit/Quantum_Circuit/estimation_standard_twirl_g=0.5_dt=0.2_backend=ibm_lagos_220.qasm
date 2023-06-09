OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4720[3];
rz(2.0128073) q[0];
sx q[0];
rz(-1.8682171) q[0];
sx q[0];
rz(1.3265106) q[0];
rz(1.3995124) q[1];
sx q[1];
rz(5.9935) q[1];
sx q[1];
rz(14.129451) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(1.1946902) q[2];
sx q[2];
rz(-0.21393531) q[2];
sx q[2];
rz(-3.1287586) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.815082) q[0];
sx q[0];
rz(-1.2733755) q[0];
sx q[0];
rz(1.1287853) q[0];
rz(-0.012834013) q[1];
sx q[1];
rz(0.21393531) q[1];
sx q[1];
rz(1.5630803) q[2];
sx q[2];
rz(-2.8519074) q[2];
sx q[2];
rz(1.7420802) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4720[0];
measure q[1] -> c4720[1];
measure q[2] -> c4720[2];
