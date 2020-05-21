using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Diagnostics;
using System.Data.SqlClient;
using Gurobi;
using System.Threading.Tasks;
using System.Threading;


namespace CMP_v1
{
    class Program
    {
        static int N, E, K, R, I;
        static int arc_weights_ON, is_BUDGETED, SAA_M, BUDGET, SAA_N2, SAA_N3;
        static int run_MIP, run_pipage, run_LR, runLT, run_reduction, run_MEM, run_CELF, save_model, save_log2;
        static int no_of_LR_iter;
        static List<ReadData> unique_arcs;
        static double fixed_probability;
        static List<string> neigh_set, arc_set, arcID_set, pred_set, neigh_edge_set;
        static List<UInt16> node_set, init_set, CA_set, active_set, NA_set, selected_set, result_set, SAA_current_list, selected_set2;
        static double[] recruitment_cost;
        static double[] SA_1_obj, y_obj, SA_1_test_obj;
        static List<List<List<UInt16>>> SAA_neigh_list, SAA_pred_list, SAA_tree_list, SAA_fwd_tree_list;
        static UInt16[,] arcs_int;
        static UInt16[] node_index;
        static List<List<UInt16>> SAA_list, SAA_pipage_list, SAA_LR_list;
        static List<UInt16> initial_infected;
        static Stopwatch sw;
        static TimeSpan sw_elapsed;

        static void Main(string[] args)
        {
            set_parameters();
            read_data();
            initialize_random_arcs();
            intialize_infected_set();
            Solver();
            Console.WriteLine("Finished!");
        }

        public static void set_parameters()
        {
            K = 5;
            R = 10;
            I = 20;
            fixed_probability = 0.1;
            SAA_M = 1; SAA_N2 = 0; SAA_N3 = 0;

            run_MIP = 1;
            arc_weights_ON = 0;
            is_BUDGETED = 0;
        }

        public static void read_data()
        {
            string path,filename;
            filename = "";
            //path = @"C:\Users\Evren\Downloads\inst\Email-Enron.im"; // "Email-Enron.im","phy.im","p2p-Gnutella04.im","CollegeMsg.im","Slashdot0902.txt"
            //path=@"C:\Users\Evren\Downloads\inst\phy.im";
            //path = @"C:\Users\Evren\Downloads\inst\p2p-Gnutella04.im";
            path = @"C:\Users\Evren\Downloads\inst\CollegeMsg.im";
            //path = @"C:\Users\Evren\Downloads\inst\Slashdot0902.txt";
            //path = @"C:\Users\Evren\Downloads\inst\" + filename;
            //path = @"D:\Influence-marketing\Code\influencer-1\weighted_ic\" + filename + ".txt";
            //path = @"D:\zz_doktora\ourpapers\2020\naoto\" + filename + ".txt";
            //
            //path=@"D:\zz_doktora\ourpapers\2017\IMP-Linear\dataset\facebook_combined.txt";
            //path = @"D:\Influence-marketing\Code\influencer-1\sample-data-4w.txt";
            //path = (@"D:\Influence-marketing\Code\influencer-1\contagion_sample_20_edges.txt");
            //path = (@"D:\Influence-marketing\Code\influencer-1\sample-data-20.txt");

            //string[] lines = System.IO.File.ReadAllLines(@"d:\Influence-marketing\Code\influencer-1\" + filename + ".txt");
            //string[] lines = System.IO.File.ReadAllLines(@"C:\cygwin64\home\Evren\IMM\" + filename + "\\graph_ic.inf");
            //path = @"D:\zz_doktora\ourpapers\2017\IMP-Linear\dataset\ruthmaier_socnet\" + filename + ".txt";
            

            var readData = (File.ReadLines(path).Skip(4).Select(a => {
                //string[] nodes_string_arr = a.Split(' ');
                //var temparc = nodes_string_arr[0] + "," + nodes_string_arr[1];
                //return new { Head = int.Parse(nodes_string_arr[0]), Tail = int.Parse(nodes_string_arr[1]),Id = a };
                return new { Id = a };
            }).AsParallel().ToList());

            if (arc_weights_ON == 0)
            {
                unique_arcs = readData.GroupBy(a => a.Id).Select(a => new { Key = a, Count = a.Count() }).Select(a =>
            new ReadData
            {
                Key = new { Head = int.Parse(a.Key.Key.Split(' ')[0]), Tail = int.Parse(a.Key.Key.Split(' ')[1]) },
                Count = a.Count,
                W = (a.Count == 1) ? fixed_probability : (1 - Math.Pow((1 - fixed_probability), a.Count)),
            }).AsParallel().ToList();
            }

            else
            {
                unique_arcs = readData.GroupBy(a => a.Id).Select(a => new { Key = a, Count = a.Count() }).Select(a =>
                new ReadData
                {
                    Key = new { Head = int.Parse(a.Key.Key.Split(' ')[0]), Tail = int.Parse(a.Key.Key.Split(' ')[1]) },
                    Count = a.Count,
                    W = double.Parse(a.Key.Key.Split(' ')[2]),
                }).AsParallel().ToList();
            }

            

            //new ReadData
            //{
            //    Key = new { Head = int.Parse(a.Key.Key.Split(',')[0]), Tail = int.Parse(a.Key.Key.Split(',')[1]) },
            //    Count = a.Count,
            //    W = Double.Parse(a.Key.Key.Split(',')[2]),
            //}).AsParallel().ToList();

            var nodes = unique_arcs.Select(a => a.Key.Head).Distinct().Union(unique_arcs.Select(b => b.Key.Tail).Distinct()).Distinct().AsParallel().ToList();
            //var left_nodes_neighbours = unique_arcs.GroupBy(a => a.Key.Head).Select(a => new { Head = a.Key, RefList = a.Select(b => b.Key.Tail).ToList() }).AsParallel().ToList();


            //List<ReadData> UniqueArcs = new List<ReadData>();
            //UniqueArcs.AddRange(unique_arcs);
            //UniqueArcs = unique_arcs.Select(a => a).AsParallel().ToList();


            arc_set = new List<string>();
            N = nodes.Count;
            E = unique_arcs.Count;
            node_set = new List<UInt16>();
            for (int i = 0; i < N; i++)
                node_set.Add((UInt16)nodes[i]);

            recruitment_cost = new double[N];
            if (is_BUDGETED == 1)
            {
                string[] lines_budget = System.IO.File.ReadAllLines(@"d:\Influence-marketing\Code\influencer-1\budget-10k.txt");
                for (int i = 0; i < N; i++)
                {
                    recruitment_cost[i] = System.Convert.ToDouble(lines_budget[i]);
                }
            }
            else
            {
                for (int i = 0; i < N; i++)
                {
                    recruitment_cost[i] = 1;
                }

            }
        }

        public static void initialize_random_arcs()
        {
            var reslist = new List<dynamic>();
            var tasks = new List<Task<List<BaseModel>>>();
            var rand = new Random();
            for (int i = 0; i < E; i++)
            {
                unique_arcs[i].Prop = rand.NextDouble();
            }

            for (int i = 0; i < R; i++)
            {
                var luckyarcs = Task.Run<List<BaseModel>>(() => unique_arcs.Select(a => a).Where(a => (a.W >= a.GetProbablity())).AsParallel().ToList().Select(b =>
                            new BaseModel { Head = b.Key.Head, Tail = b.Key.Tail }).AsParallel().ToList());
                tasks.Add(luckyarcs);
                //var luckyarcs = unique_arcs.Select(a => a).Where(a => (a.W >= GetUniform())).AsParallel().ToList().Select(b =>
                //               new BaseModel { Head = b.Key.Head, Tail = b.Key.Tail }).AsParallel().ToList();
                //reslist.Add(luckyarcs);
            }
            var results = Task.WhenAll(tasks);
            foreach (var task in tasks)
            {
                var l1 = task.Result;
                reslist.Add(l1);
            }
            initialize_SAA_neighbourhood_fast(R, reslist);
        }

        public static void initialize_SAA_neighbourhood_fast(int sample_size, List<dynamic> reslist)
        {
            System.Console.WriteLine("Starting Fast SAA_Neigh");
            SAA_neigh_list = new List<List<List<UInt16>>>();
            SAA_pred_list = new List<List<List<UInt16>>>();
            int counter2 = 0;
            //x_exist = new bool[sample_size, N];

            int Nmax = node_set.Max() + 1;
            node_index = new UInt16[Nmax];
            for (UInt16 i = 0; i < N; i++)
            {
                node_index[node_set[i]] = i;
            }

            for (int r = 0; r < R; r++)
            {
                SAA_neigh_list.Add(new List<List<UInt16>>());
                SAA_pred_list.Add(new List<List<UInt16>>());

                for (UInt16 i = 0; i < N; i++)
                {
                    //x_exist[i, r] = true;
                    SAA_neigh_list[r].Add(new List<UInt16>());
                    SAA_pred_list[r].Add(new List<UInt16>());
                }
            }

            UInt16 head, tail;

            for (int r = 0; r < sample_size; r++)   //Parallel.For(0, sample_size, r =>
            {
                foreach (BaseModel item in reslist[r])
                {
                    head = node_index[(UInt16)item.Head];
                    tail = node_index[(UInt16)item.Tail];
                    SAA_neigh_list[r][head].Add(tail);
                    SAA_pred_list[r][tail].Add(head);
                    counter2++;
                }
            }
            //);
            //swtemp.Close();
            System.Console.WriteLine("1-Finished Fast Neigh! ...... Counter:" + E * R + " & Counter2:" + counter2);
            //determine_network_trees_IM(sample_size, m_list);
        }  //create stochastic neighbourhood list for each node, for each sample, for each batch

        public static void intialize_infected_set()
        {
            Random rnd = new Random();
            int random_node = 0;
            initial_infected = new List<ushort>();
            //initial_infected.Add(0);            initial_infected.Add(4);            initial_infected.Add(7);
            for (int i=0; i<I;i++)
            {
                random_node = rnd.Next(N);
                if(initial_infected.IndexOf((UInt16)random_node)<0)
                    initial_infected.Add((UInt16)random_node);
                else
                {
                    i = i - 1;
                }
            }
        }


        public static void Solver()
        {
            
            //trial_limit = SAA_N1;
            SA_1_obj = new double[SAA_M];
            y_obj = new double[N];

            double max_obj = -1;
            //int type = 1; //SAA'ya gir-1
            SA_1_test_obj = new double[SAA_M];
            SAA_list = new List<List<UInt16>>(SAA_M);  
            SAA_pipage_list = new List<List<UInt16>>(SAA_M);
            SAA_LR_list = new List<List<UInt16>>(SAA_M);
            SAA_tree_list = new List<List<List<ushort>>>();
            string SAA_str_solution = "";
            Double dt_IP = 0; Double dt_PipageP = 0; Double dt_greedy = 0; Double dt_LR = 0;
            int SAA_T=0;
            double SA_1_hat_obj = 0; double R1_avg, R1_var, R1_std; 
            double[] SA_2_obj, T_LB;
            int k;
            double T2_avg = 0; double T2_var = 0; double T2_std = 0; double SA_3_obj; SA_3_obj = 0; R1_std = 0;

            double[] LP_arr = new double[SAA_M]; double[] LR_UB_arr = new double[SAA_M];
            double[] IP_arr = new double[SAA_M]; double[] LR_LB_arr = new double[SAA_M];
            double[] LP_inf = new double[SAA_M]; double[] LR_inf_arr = new double[SAA_M];
            string[] sol_arr = new string[SAA_M]; string[] LR_sol_arr = new string[SAA_M];
            double tot_LP = 0; double tot_IP = 0; double avg_LP = 0; double avg_IP = 0; double best_IP = 0; int best_index_LP = -1;
            double LR_tot_UB = 0; double LR_tot_LB = 0; double LR_avg_UB = 0; double LR_avg_LB = 0; double LR_best_inf = 0; int LR_best_index = -1; string LR_best_seed = "";
            double LR_UB_std = 0; double t_LR=0;
            double LR_inf_std = 0;
            double LR_total_inf = 0;

            List<UInt16> temp_list = new List<UInt16>();
            
            if (run_MIP == 1 ||  run_LR == 1)
            {
                Stopwatch sw_LR = new Stopwatch(); Stopwatch swx = new Stopwatch(); Stopwatch sw = new Stopwatch();
                double temp_obj = 0;
                int max_index = -1;
                swx.Start();
                for (int i = 0; i < SAA_M; i++)
                {

                    //if (runLT == 1)
                    //    runLT = 1;
                    ////initialize_random_arcs_LT(SAA_N1);
                    //else
                    //    initialize_random_arcs();

                    if (run_MIP >0 || run_reduction == 1 || run_LR == 1)
                    {
                        //if (run_MEM == 1)
                        //determine_network_trees_mem(SAA_N1);
                        //else
                        //determine_network_trees(SAA_N1);
                    }
                    swx.Stop(); System.Console.WriteLine(swx.Elapsed.Ticks); swx.Reset();
                    
                    //if (run_reduction == 1)
                    //    reduce_tree_list();

                    if (run_MIP == 1)
                    {
                        sw.Start();
                        System.Console.WriteLine("Solving " + (i + 1) + "-th SAA problem");
                        SA_1_obj[i] = mip_model(i, 1, R);
                        SA_1_obj[i] = mip_model(i, 2, R);
                        temp_obj = temp_obj + SA_1_obj[i];
                        sw.Stop();
                        System.Console.WriteLine(sw.ElapsedTicks);
                        dt_IP = sw.ElapsedMilliseconds;
                    }


                    if (run_LR == 1)
                    {
                        sw_LR.Start();
                        //z_LRUB is written in function, z_LRLB is what function returns (feasible solution), zLRinf, t_LR
                        //initialize_SAA_neighbourhood(SAA_N1); //determine_network_trees(SAA_N1);
                        //theoption = 1;  determine_network_trees(SAA_N1);
                        //determine_fwd_network_trees_list(SAA_N1);
                        //double initialize_duals= (double)construct_model2_gurobi_LP(i, 0, 1, 0);
                        //LR_LB_arr[i] = (double)LR_3_Shabbir(i, 0, SAA_N1) / SAA_N1;
                        //determine_network_trees_mem(SAA_N1);
                        //if (run_MEM == 1)
                        //  LR_LB_arr[i] = (double)LR_0_mem(i, 0, SAA_N1) / SAA_N1;
                        //else

                        //LR için burayı aç 26/04/2020
                        //LR_LB_arr[i] = (double)LR_0(i, 0, R) / R;

                        //double LR2_result= (double)LR_0(i, 1, SAA_N1) / SAA_N1;
                        //if (LR_LB_arr[i]> LR2_result)
                        //{
                        //    System.Console.WriteLine("LR1 : " + LR_LB_arr[i] + " LR2 : +" + LR2_result);
                        //}



                        //myoption = 0; LR_LB_arr[i] = (double)LR_0(i, 1, SAA_N1) / SAA_N1;

                        //reduce_tree_list();
                        //theoption = 1;
                        //swx.Start();
                        //determine_network_trees(SAA_N1); swx.Stop(); System.Console.WriteLine(swx.Elapsed.Ticks); swx.Reset();
                        //LR_LB_arr[i] = (double)LR_0(i, 1, SAA_N1) / SAA_N1;

                        //swx.Start(); determine_network_trees_mem(SAA_N1); swx.Stop(); System.Console.WriteLine(swx.Elapsed.Ticks); swx.Reset();
                        //LR_LB_arr[i] = (double)LR_0_mem(i, 1, SAA_N1) / SAA_N1;
                        //LR_LB_arr[i] = (double)LR_0(i, 1, SAA_N1) / SAA_N1;

                        //LR için burayı aç 26/04/2020
                        //LR_UB_arr[i] = z_LRUB;
                        //LR_sol_arr[i] = solution_LR;
                        sw_LR.Stop();
                        double sw_LR_elapsed = sw_LR.ElapsedMilliseconds; t_LR = sw_LR_elapsed;
                    }

                }


                //reporting

                if (run_MIP == 1)
                {
                    sw.Reset(); sw.Start();
                    SA_1_hat_obj = temp_obj / SAA_M;

                    R1_avg = SA_1_hat_obj;
                    R1_var = 0;
                    for (int i = 0; i < SAA_M; i++)
                    {
                        R1_var = R1_var + (R1_avg - SA_1_obj[i]) * (R1_avg - SA_1_obj[i]);
                    }
                    R1_var = R1_var / (SAA_M - 1);
                    R1_std = Math.Sqrt(R1_var);

                    string[] stringsArray;
                    string the_vars_list;

                    temp_obj = 0;
                    System.Console.WriteLine("SAA upperbund : " + SA_1_hat_obj);
                    //SA_2_obj = new double[SAA_M];
                    ////SAA-2'leri LP olarak çözeceksen bu iki satırı açman gerek
                    ////initialize_SAA_neighbourhood(SAA_M, SAA_N2);
                    ////determine_network_depth(SAA_N2);
                    //T_LB = new double[SAA_M];
                    //for (int i = 0; i < SAA_M; i++)
                    //{
                    //    stringsArray = SAA_list[i].Select(the_i => the_i.ToString()).ToArray();
                    //    the_vars_list = string.Join(",", stringsArray);

                    //    T_LB[i] = 0;

                    //    for (int j = 0; j < SAA_T; j++)
                    //    {
                    //        ////initialize_SAA_2_prb(i * SAA_M + j);
                    //        SAA_current_list = new List<UInt16>(SAA_list[i]);
                    //        //temp_obj = evaluate_influence(SAA_list[i], i, SAA_N2, 1);
                    //        T_LB[i] = T_LB[i] + temp_obj;
                    //        System.Console.WriteLine("Evaluating x: " + the_vars_list + ", and " + (i + 1) + "-th objective (" + SA_1_obj[i] + ") -->" + T_LB[i]);
                    //    }
                    //    T_LB[i] = T_LB[i] / SAA_T;


                    //    if (T_LB[i] > max_obj)
                    //    {
                    //        max_obj = T_LB[i];
                    //        max_index = i;
                    //    }
                    //    SA_2_obj[i] = T_LB[i];
                    //    //temp_obj = mip_model(i, 1, SAA_N2); ;
                    //    //SAA_list_string = string.Join(",", );

                    //}

                    //max_obj = SA_2_obj[max_index];
                    //System.Console.WriteLine("Re-Evaluating " + (max_index + 1) + "-th objective with influence : " + max_obj);


                    //// STEP - 3 OF SAA METHOD

                    //double[] T2_LB = new double[SAA_T];
                    //for (int j = 0; j < SAA_T; j++)
                    //{
                    //    ////initialize_SAA_3_prb();
                    //    //T2_LB[j] = evaluate_influence(SAA_list[max_index], 0, SAA_N3, 2);
                    //    T2_avg = T2_avg + T2_LB[j];
                    //}
                    //T2_avg = T2_avg / SAA_T;
                    //T2_var = 0;
                    //for (int j = 0; j < SAA_T; j++)
                    //{
                    //    T2_var = T2_var + (T2_avg - T2_LB[j]) * (T2_avg - T2_LB[j]);
                    //}
                    //T2_var = T2_var / (SAA_T - 1);
                    //T2_std = Math.Sqrt(T2_var);

                    //SA_3_obj = T2_avg;


                    ////UInt16[] st_arr = SAA_list[max_index].ToArray();
                    max_index = 0;
                    k = SAA_list[max_index].Count;
                    SAA_list[max_index].Sort();
                    //string SAA_str_solution = "";
                    for (int i = 0; i < k; i++)
                    {
                        //sww.WriteLine(result_set[i]);
                        SAA_str_solution = SAA_str_solution + "," + node_set[(int)SAA_list[max_index][i]];
                    }

                    sw.Stop(); dt_IP = dt_IP + sw.ElapsedMilliseconds;

                    System.Console.WriteLine("-------------summary---------------");
                    System.Console.WriteLine("SA_1_hat : " + SA_1_hat_obj);
                    System.Console.WriteLine("SA_2* : " + max_obj + " the " + (max_index + 1) + "-th sample");
                    System.Console.WriteLine("SA_3* : " + SA_3_obj);
                    System.Console.WriteLine("Optimality Gap : " + (SA_1_hat_obj - SA_3_obj) + " (% " + 100 * (SA_1_hat_obj - SA_3_obj) / SA_3_obj + ")");
                    System.Console.WriteLine("Time : " + System.Convert.ToDouble(dt_IP / 10000000.0));
                    //swvar.WriteLine("SA_1_hat : " + SA_1_hat_obj);
                    //swvar.WriteLine("SA_2* : " + max_obj + " the " + (max_index + 1) + "-th sample");
                    //swvar.WriteLine("SA_3* : " + SA_3_obj);
                    //swvar.WriteLine("Optimality Gap : " + (SA_1_hat_obj - SA_3_obj) + " (% " + 100 * (SA_1_hat_obj - SA_3_obj) / SA_3_obj + ")");
                }


                
                //LR results

                //LR results end

                sw.Reset(); sw_LR.Reset();
            }


           
            string sqltext;
            string constr = "Server=localhost\\SQLExpress;Database=research;User Id=doktora;Password=cesmede;";
            SqlConnection conn = new SqlConnection(constr);
            conn.Open();
            SqlCommand command = conn.CreateCommand();

            

            {
                sqltext = "INSERT INTO BIMP_SAA (SAA_M,SAA_N1,SAA_N2,SAA_N3,modelID, diffusion_modelID,K,network_sampleID,node_size,edge_size,duration,z_star,SAA_z, SAA_z2, SAA_z3,solution_y, propagation_prop, budget,R1_std, T2_std,z_pipageLP,z_pipageIP,z_pipageInf,t_pipage,pipage_solution,pipage_count, z_LRUB, z_LRLB, z_LRinf,t_LR, solution_LR, stdev_LRUB, stdev_LRinf, fixed_p, LR_iter) VALUES (" + SAA_M + "," + R + "," + SAA_N2 + "," + SAA_N3 + ",1,1," + K + ",1," + N + "," + E + "," + dt_IP + ",-1," + SA_1_hat_obj + "," + max_obj + "," + SA_3_obj + ",'" + SAA_str_solution + "'," + 0 + ",'" + BUDGET + "','" + R1_std + "','" + T2_std + "'," + 0 + "," + 0 + "," + 0 + "," + 0 + ",'',''," + LR_avg_UB + "," + LR_avg_LB + "," + LR_best_inf + "," + t_LR + ",'" + LR_best_seed + "'," + LR_UB_std + "," + LR_inf_std + "," + fixed_probability + "," + no_of_LR_iter + ")";
                sqltext = sqltext.Replace("'NaN'", "'0'");
                command.CommandText = sqltext;
                //command.ExecuteNonQuery();
            }

            command.Dispose();
            conn.Close();
            if(SAA_tree_list.Count>0)
                SAA_tree_list.Clear();
            //SAA_fwd_tree_list.Clear();
        }

        public static double mip_model(int SAA_m, int SAA_type, int sample_size)
        {
            
            save_model = 0; // to save model.lp file
            save_log2 = 0;  // to save the solution of model
            double influence = 0;

            Stopwatch st1 = new Stopwatch();

            st1.Start();
            if (run_MIP == 1)
            {
                st1.Start();
                if (SAA_type == 1)
                {                //Gurobi8.0
                    influence = (double)MIP_node_based(SAA_m, SAA_type, sample_size) / sample_size;
                }

                if (SAA_type == 2)
                {                //Gurobi8.0
                    influence = (double)MIP_arc_based(SAA_m, SAA_type, sample_size) / sample_size;
                }
                System.Console.WriteLine("Integer sol: " + influence);

            }

           
            System.Console.WriteLine(st1.Elapsed);
            st1.Reset();

            // System.Console.ReadKey();
            return influence;
        }


        public static double MIP_node_based(int SAA_m, int SAA_type, int sample_size)
        {
            int trial_limit = sample_size;
            System.Console.WriteLine("Constructing the MIP with Gurobi...");
            double solution = -1;
            try
            {
                GRBEnv env = new GRBEnv("mip1.log");
                GRBModel model = new GRBModel(env);
                //model.Parameters.Presolve = 1;

                GRBVar[] y_j = new GRBVar[N];
                GRBVar[,] x_ir = new GRBVar[N, sample_size];

                System.Console.WriteLine("Creating y_i");
                for (int i = 0; i < N; i++)
                {
                    y_j[i] = model.AddVar(0.0, 1.0, y_obj[i], GRB.BINARY, "y_" + i);
                }

                System.Console.WriteLine("Creating x_ir");
                for (int i = 0; i < N; i++)
                  for (int r = 0; r < sample_size; r++)
                {
                    x_ir[i, r] = model.AddVar(0.0, 1.0, 1.0, GRB.CONTINUOUS, "x_" + i + "_" + r);
                }

                model.ModelSense = GRB.MINIMIZE;

                //----------------------------------------------------------
                //--------------- create the constraints
                int counter = 0;
                GRBLinExpr temp_exp2;
                temp_exp2 = 0.0;
                System.Console.WriteLine("Starting the constraints... Total : " + (trial_limit * N + 1) + " constraints");
                // exactly k initial active users
                for (int i = 0; i < N; i++)
                {
                    temp_exp2.AddTerm(1.0, y_j[i]);
                }

                model.AddConstr(temp_exp2 == K, "constraint_y");
                temp_exp2 = 0.0;
                //--- influence constraints x_i_r <= Sum_j (y_j)          j in all accessing nodes to i          (total of N.R constraints)
                //string[] neigh_arr;


                GRBLinExpr temp_exp;

                int j = 0;

                temp_exp = 0.0;
                int counter2 = 0;

                for (int r = 0; r < trial_limit; r++)
                {
                    for (int i = 0; i < N; i++)
                    {
                        //if (SAA_tree_list[r][i].Count > 0)
                        //if (x_exist[r, i] == true)
                        {
                            //x_ir[i, r] = model.AddVar(0.0, 1.0, 1.0, GRB.CONTINUOUS, "x_" + i + "_" + r);
                            temp_exp.AddTerm(1.0, x_ir[i, r]);
                            temp_exp.AddTerm(1.0, y_j[i]);

                            //if (SAA_tree_list[r][i].Count <= N)
                            {
                                foreach (UInt16 node in SAA_pred_list[r][i])   //my predecessors can activate me
                                {
                                    {
                                        temp_exp2.AddTerm(-1, x_ir[(int)node,r]);
                                        model.AddConstr(temp_exp+temp_exp2 >= 0, "constraint_2" + (counter + 1));
                                        temp_exp2 = 0.0;
                                        counter++;
                                    }
                                }
                            }
                        }       //end of if for null neighbourhood
                        temp_exp = 0.0;
                    }           //end of for loop for nodes
                }               //end of for loop for sample size R

                for (int r = 0; r < trial_limit; r++)
                {
                    for (int i = 0; i < I; i++)
                    {
                        temp_exp.AddTerm(1.0, x_ir[(int)initial_infected[i], r]);
                        model.AddConstr(temp_exp >= 1, "constraint_3" + (counter + 1));
                        temp_exp = 0.0;
                    }
                }
                for (int i = 0; i < I; i++)
                {
                    temp_exp.AddTerm(1.0, y_j[(int)initial_infected[i]]);
                    model.AddConstr(temp_exp == 0, "constraint_3" + (counter + 1));
                    temp_exp = 0.0;
                }

                model.Write("model.lp");
                //model.Write("modelGRB2" + SAA_m + ".mps");
                //model.Write("modelGRB2" + SAA_m + ".lp");
                GRBModel p = model.Presolve();
                p.Write("presolve.lp");

                model.Optimize();

                if (model.Status == GRB.Status.OPTIMAL)
                {
                    Console.WriteLine("Obj: " + model.Get(GRB.DoubleAttr.ObjVal));
                    solution = model.Get(GRB.DoubleAttr.ObjVal);


                    List<UInt16> SAA_sample_result = new List<UInt16>(K);
                    int isfractional = 0;
                    result_set = new List<ushort>();
                    for (int jj = 0; jj < N; ++jj)
                    {

                        if (y_j[jj].X > 0.001)
                        {
                            //result_set.Add(node_set[jj]);
                            result_set.Add(node_set[jj]);
                            SAA_sample_result.Add((UInt16)jj);
                            System.Console.WriteLine(y_j[jj].VarName + "=" + y_j[jj].X);
                            if (y_j[jj].X < 0.9)
                            {
                                System.Console.WriteLine("Fractional value found");
                                System.Console.ReadKey();
                                isfractional = 1;
                            }
                        }
                    }
                    if (isfractional == 1)
                    {
                        System.Console.WriteLine("To conitnue click...");
                        System.Console.ReadKey();
                    }

                    SAA_list.Add(SAA_sample_result);


                }
                else
                {
                    Console.WriteLine("No solution");
                    solution = 0;
                }

                // Dispose of model and env

                model.Dispose();
                env.Dispose();


            }
            catch (GRBException e)
            {
                Console.WriteLine("Error code: " + e.ErrorCode + ". " + e.Message);
            }


            return solution;
        }


        public static double MIP_arc_based(int SAA_m, int SAA_type, int sample_size)
        {
            int trial_limit = sample_size;
            System.Console.WriteLine("Constructing the MIP with Gurobi...");
            double solution = -1;
            try
            {
                GRBEnv env = new GRBEnv("mip1.log");
                GRBModel model = new GRBModel(env);
                //model.Parameters.Presolve = 1;

                GRBVar[] y_j = new GRBVar[E];
                GRBVar[,] x_ir = new GRBVar[N, sample_size];

                System.Console.WriteLine("Creating y_i (edges)");
                for (int i = 0; i < E; i++)
                {
                    y_j[i] = model.AddVar(0.0, 1.0, 0, GRB.BINARY, "y_" + node_index[unique_arcs[i].Key.Head]+"_"+ node_index[unique_arcs[i].Key.Tail]);
                }

                System.Console.WriteLine("Creating x_ir");
                for (int i = 0; i < N; i++)
                    for (int r = 0; r < sample_size; r++)
                    {
                        x_ir[i, r] = model.AddVar(0.0, 1.0, 1.0, GRB.CONTINUOUS, "x_" + i + "_" + r);
                    }

                model.ModelSense = GRB.MINIMIZE;

                //----------------------------------------------------------
                //--------------- create the constraints
                int counter = 0;
                GRBLinExpr temp_exp2;
                temp_exp2 = 0.0;
                System.Console.WriteLine("Starting the constraints... Total : " + (trial_limit * N + 1) + " constraints");
                // exactly k initial active users
                for (int i = 0; i < E; i++)
                {
                    temp_exp2.AddTerm(1.0, y_j[i]);
                }

                model.AddConstr(temp_exp2 == K, "constraint_y");
                temp_exp2 = 0.0;
                //--- influence constraints x_i_r <= Sum_j (y_j)          j in all accessing nodes to i          (total of N.R constraints)
                //string[] neigh_arr;


                GRBLinExpr temp_exp;

                int j = 0;

                temp_exp = 0.0;
                int counter2 = 0;
                ReadData temp_arc = new ReadData();

                for (int r = 0; r < trial_limit; r++)
                {
                    for (int i = 0; i < N; i++)
                    {
                        //if (SAA_tree_list[r][i].Count > 0)
                        //if (x_exist[r, i] == true)
                        {
                            //x_ir[i, r] = model.AddVar(0.0, 1.0, 1.0, GRB.CONTINUOUS, "x_" + i + "_" + r);
                            temp_exp.AddTerm(1.0, x_ir[i, r]);

                            //if (SAA_tree_list[r][i].Count <= N)
                            {
                                foreach (UInt16 node in SAA_pred_list[r][i])   //my predecessors can activate me
                                {
                                    var x = unique_arcs.FindIndex(a => a.Key.Head == node_set[(int)node] && a.Key.Tail == node_set[i]);
                                    //int arcID=unique_arcs.IndexOf()
                                    temp_exp2.AddTerm(-1, x_ir[(int)node, r]);
                                    temp_exp2.AddTerm(1, y_j[x]);
                                    model.AddConstr(temp_exp + temp_exp2 >= 0, "constraint_2" + (counter + 1));
                                    temp_exp2 = 0.0;
                                    counter++;
                                }
                            }
                        }       //end of if for null neighbourhood
                        temp_exp = 0.0;
                    }           //end of for loop for nodes
                }               //end of for loop for sample size R

                for (int r = 0; r < trial_limit; r++)
                {
                    for (int i = 0; i < I; i++)
                    {
                        temp_exp.AddTerm(1.0, x_ir[(int)initial_infected[i], r]);
                        model.AddConstr(temp_exp >= 1, "constraint_3" + (counter + 1));
                        temp_exp = 0.0;
                    }
                }
                //no need for this in edge blocking
                //for (int i = 0; i < I; i++)
                //{
                //    temp_exp.AddTerm(1.0, y_j[(int)initial_infected[i]]);
                //    model.AddConstr(temp_exp == 0, "constraint_3" + (counter + 1));
                //    temp_exp = 0.0;
                //}

                model.Write("model.lp");
                //model.Write("modelGRB2" + SAA_m + ".mps");
                //model.Write("modelGRB2" + SAA_m + ".lp");
                GRBModel p = model.Presolve();
                p.Write("presolve.lp");

                model.Optimize();

                if (model.Status == GRB.Status.OPTIMAL)
                {
                    Console.WriteLine("Obj: " + model.Get(GRB.DoubleAttr.ObjVal));
                    solution = model.Get(GRB.DoubleAttr.ObjVal);


                    List<UInt16> SAA_sample_result = new List<UInt16>(K);
                    int isfractional = 0;
                    result_set = new List<ushort>();
                    for (int jj = 0; jj < y_j.Count(); ++jj)
                    {

                        if (y_j[jj].X > 0.001)
                        {
                            //result_set.Add(node_set[jj]);
                            result_set.Add(node_set[jj]);
                            SAA_sample_result.Add((UInt16)jj);
                            System.Console.WriteLine(y_j[jj].VarName + "=" + y_j[jj].X);
                            if (y_j[jj].X < 0.9)
                            {
                                System.Console.WriteLine("Fractional value found");
                                System.Console.ReadKey();
                                isfractional = 1;
                            }
                        }
                    }
                    if (isfractional == 1)
                    {
                        System.Console.WriteLine("To conitnue click...");
                        System.Console.ReadKey();
                    }

                    SAA_list.Add(SAA_sample_result);


                }
                else
                {
                    Console.WriteLine("No solution");
                    solution = 0;
                }

                // Dispose of model and env

                model.Dispose();
                env.Dispose();


            }
            catch (GRBException e)
            {
                Console.WriteLine("Error code: " + e.ErrorCode + ". " + e.Message);
            }


            return solution;
        }

        public static double LR_node()
        {
            double[] L = new double[N];
            return 1;
        }

    }
}
