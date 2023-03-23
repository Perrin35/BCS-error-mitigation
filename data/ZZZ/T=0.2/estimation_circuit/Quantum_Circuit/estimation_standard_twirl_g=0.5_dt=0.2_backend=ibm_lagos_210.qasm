OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4710[3];
rz(5.7515563) q[0];
sx q[0];
rz(3.2885056) q[0];
sx q[0];
rz(11.983903) q[0];
rz(-0.12162265) q[1];
sx q[1];
rz(-0.096225503) q[1];
sx q[1];
rz(2.7715575) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
rz(-1.5680383) q[2];
sx q[2];
rz(-1.843455) q[2];
sx q[2];
rz(0.83525036) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.5591247) q[0];
sx q[0];
rz(-2.9946798) q[0];
sx q[0];
rz(0.53162901) q[0];
rz(-0.83525036) q[1];
sx q[1];
rz(-1.843455) q[1];
sx q[1];
rz(-0.37003511) q[2];
sx q[2];
rz(-3.0453672) q[2];
sx q[2];
rz(-3.01997) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4710[0];
measure q[1] -> c4710[1];
measure q[2] -> c4710[2];
