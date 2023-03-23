OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4668[3];
rz(-0.17488527) q[0];
sx q[0];
rz(-1.2198679) q[0];
sx q[0];
rz(0.81982293) q[0];
rz(-1.6490844) q[1];
sx q[1];
rz(-1.9928056) q[1];
sx q[1];
rz(-1.2951712) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.1621651) q[2];
sx q[2];
rz(-2.6178306) q[2];
sx q[2];
rz(-1.8996138) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.3217697) q[0];
sx q[0];
rz(-1.9217247) q[0];
sx q[0];
rz(-2.9667074) q[0];
rz(-1.8996138) q[1];
sx q[1];
rz(-0.52376206) q[1];
sx q[1];
rz(-1.8464214) q[2];
sx q[2];
rz(-1.9928056) q[2];
sx q[2];
rz(1.6490844) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4668[0];
measure q[1] -> c4668[1];
measure q[2] -> c4668[2];