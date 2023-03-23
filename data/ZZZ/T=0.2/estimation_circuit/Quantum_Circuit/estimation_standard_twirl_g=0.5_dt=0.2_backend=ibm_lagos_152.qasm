OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4652[3];
rz(1.4289186) q[0];
sx q[0];
rz(-0.83587712) q[0];
sx q[0];
rz(0.32444251) q[0];
rz(-1.1724053) q[1];
sx q[1];
rz(-0.3243341) q[1];
sx q[1];
rz(2.2066051) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(-1.8040498) q[2];
sx q[2];
rz(-0.62061824) q[2];
sx q[2];
rz(-0.56819714) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.32444251) q[0];
sx q[0];
rz(-0.83587712) q[0];
sx q[0];
rz(-1.4289186) q[0];
rz(2.5733955) q[1];
sx q[1];
rz(2.5209744) q[1];
sx q[1];
rz(-0.9349876) q[2];
sx q[2];
rz(-2.8172586) q[2];
sx q[2];
rz(-1.9691874) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4652[0];
measure q[1] -> c4652[1];
measure q[2] -> c4652[2];
