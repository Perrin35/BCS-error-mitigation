OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4588[3];
rz(1.3538395) q[0];
sx q[0];
rz(4.1551808) q[0];
sx q[0];
rz(15.566047) q[0];
rz(-0.74001835) q[1];
sx q[1];
rz(-3.0844223) q[1];
sx q[1];
rz(-1.4800164) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.9136007) q[2];
sx q[2];
rz(-1.5214464) q[2];
sx q[2];
rz(2.3625236) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.14191597) q[0];
sx q[0];
rz(-1.0135881) q[0];
sx q[0];
rz(1.7877532) q[0];
rz(2.3625236) q[1];
sx q[1];
rz(-1.6201463) q[1];
sx q[1];
rz(1.6615762) q[2];
sx q[2];
rz(-0.05717036) q[2];
sx q[2];
rz(-2.4015743) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4588[0];
measure q[1] -> c4588[1];
measure q[2] -> c4588[2];
