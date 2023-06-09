OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4544[3];
rz(3.1997147) q[0];
sx q[0];
rz(6.0877803) q[0];
sx q[0];
rz(14.182325) q[0];
rz(3.9865239) q[1];
sx q[1];
rz(4.6098843) q[1];
sx q[1];
rz(14.945324) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
rz(4.3057726) q[2];
sx q[2];
rz(5.195959) q[2];
sx q[2];
rz(12.543922) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
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
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.5256381) q[0];
sx q[0];
rz(-0.19540499) q[0];
sx q[0];
rz(3.0834706) q[0];
rz(3.1191445) q[1];
sx q[1];
rz(-2.0543663) q[1];
sx q[1];
rz(2.3789537) q[2];
sx q[2];
rz(-1.4682917) q[2];
sx q[2];
rz(-0.84493121) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4544[0];
measure q[1] -> c4544[1];
measure q[2] -> c4544[2];
