using System;
using System.Collections.Generic;
using System.Text;

namespace CMP_v1
{
    public class ReadData
    {
        public dynamic Key { get; set; } = new { };
        public double W { get; set; }
        public int Count { get; set; }
        public double Prop { get; set; } = new Random().NextDouble();
        public double GetProbablity()
        {
            Prop = new Random().NextDouble();
            return Prop;
        }
    }
}
