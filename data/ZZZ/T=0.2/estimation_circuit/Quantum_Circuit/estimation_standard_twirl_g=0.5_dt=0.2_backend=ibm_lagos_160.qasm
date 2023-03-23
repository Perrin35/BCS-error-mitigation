OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4660[3];
rz(-0.57763343) q[0];
sx q[0];
rz(-1.9413318) q[0];
sx q[0];
rz(2.9637458) q[0];
rz(-1.9260288) q[1];
sx q[1];
rz(-1.4357683) q[1];
sx q[1];
rz(1.3081843) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
rz(1.2864575) q[2];
sx q[2];
rz(-2.4830806) q[2];
sx q[2];
rz(2.1381974) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.17784684) q[0];
sx q[0];
rz(-1.9413318) q[0];
sx q[0];
rz(0.57763343) q[0];
rz(-1.0033952) q[1];
sx q[1];
rz(0.65851209) q[1];
sx q[1];
rz(1.3081843) q[2];
sx q[2];
rz(-1.7058244) q[2];
sx q[2];
rz(-1.2155638) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4660[0];
measure q[1] -> c4660[1];
measure q[2] -> c4660[2];